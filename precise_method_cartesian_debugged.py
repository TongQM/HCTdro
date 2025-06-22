import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy import optimize, integrate, linalg
from classes_cartesian import SquareRegion, Coordinate, DemandsGenerator, Polyhedron, Demand
import torch
import torchquad
import warnings
warnings.filterwarnings('ignore')


def find_worst_tsp_density_precise_debugged(region: SquareRegion, demands, t: float = 1, epsilon: float = 0.1, tol: float = 1e-4, use_torchquad: bool = True, max_iterations: int = 5):
    '''
    Debugged precise method: Analytic center cutting plane method with numerical stability improvements
    Uses smaller examples and faster integration for debugging.
    '''
    n = demands.shape[0]
    UB, LB = np.inf, -np.inf
    lambdas_bar = np.zeros(n)
    bound_value = min(region.side_length, 0.5)
    
    # Fix polyhedron initialization - ensure c is a numpy array
    A = np.eye(n)
    b = bound_value * np.ones(n)
    B = np.ones((1, n))
    c = np.array([0.0])  # Make sure c is a numpy array
    
    polyhedron = Polyhedron(A, b, B, c, n)
    k = 1
    best_lambdas = None
    best_v_tilde = None
    best_demands_locations = None
    
    print(f"Initial polyhedron bounds: {bound_value}")
    print(f"Initial polyhedron A shape: {polyhedron.A.shape}, b shape: {polyhedron.b.shape}")
    print(f"Initial polyhedron B shape: {polyhedron.B.shape}, c shape: {polyhedron.c.shape}")
    
    while (abs(UB - LB) > epsilon) and (k <= max_iterations):
        print(f'\n{"="*60}')
        print(f'Iteration {k} begins:')
        print(f'Current gap: |UB - LB| = |{UB:.6f} - {LB:.6f}| = {abs(UB - LB):.6f}')
        print(f'Target epsilon: {epsilon}')
        print(f'{"="*60}')
        starttime = time.time()
        
        try:
            # Debug analytic center
            print(f"\n--- Analytic Center Debug ---")
            
            start_time = time.time()
            lambdas_bar, _ = polyhedron.find_analytic_center_with_phase1(lambdas_bar)
            end_time = time.time()
            
            print(f"Find analytic center: Lambdas_bar is {lambdas_bar}, with value {_}, took {end_time - start_time}s.")
            
            demands_locations = np.array([d.get_coordinates() for d in demands])
            
            # Debug problem 14 optimization
            print(f"\n--- Problem 14 Debug ---")
            v_bar, problem14_func_val = minimize_problem14_cartesian_debugged(demands, lambdas_bar, t, region, use_torchquad)
            print(f"Problem 14 result: v_bar = {v_bar}, func_val = {problem14_func_val}")
            
            if use_torchquad:
                UB = compute_upper_bound_torchquad_debug(demands_locations, lambdas_bar, v_bar, region)
            else:
                UB, UB_error = compute_upper_bound_scipy(demands_locations, lambdas_bar, v_bar, region, tol)
            time2 = time.time()
            print(f'Find upper bound: Upper bound is {UB:.6f}, took {time2 - starttime}s.')
            
            # Debug problem 7 optimization
            print(f"\n--- Problem 7 Debug ---")
            v_tilde, problem7_func_val = minimize_problem7_cartesian_debugged(lambdas_bar, demands, t, region, use_torchquad)
            print(f"Problem 7 result: v_tilde = {v_tilde}, func_val = {problem7_func_val}")
            
            if use_torchquad:
                LB = compute_lower_bound_torchquad_debug(demands_locations, lambdas_bar, v_tilde, region)
            else:
                LB, LB_error = compute_lower_bound_scipy(demands_locations, lambdas_bar, v_tilde, region, tol)
            time3 = time.time()
            print(f'Find lower bound: Lower bound is {LB:.6f}, took {time3 - time2}s.')
            
            print(f"\n--- Bounds Summary ---")
            print(f"Upper Bound (UB): {UB:.6f}")
            print(f"Lower Bound (LB): {LB:.6f}")
            print(f"Gap: {abs(UB - LB):.6f}")
            
            if best_lambdas is None or LB > (best_lambdas[1] if best_lambdas[1] is not None else -np.inf):
                best_lambdas = (lambdas_bar.copy(), LB)
                best_v_tilde = v_tilde.copy()
                best_demands_locations = demands_locations.copy()
                print(f"Updated best solution with LB = {LB:.6f}")
            
            # Debug g vector computation
            print(f"\n--- G Vector Debug ---")
            g = np.zeros(len(demands))
            for i in range(len(demands)):
                if use_torchquad:
                    g[i] = compute_g_integral_torchquad_debug(i, demands_locations, lambdas_bar, v_bar, region)
                else:
                    g[i], g_error = compute_g_integral_scipy(i, demands_locations, lambdas_bar, v_bar, region, tol)
            
            g_norm = np.linalg.norm(g)
            if g_norm > 1e-6:
                g = g / g_norm
            
            constraint_value = g.T @ lambdas_bar + 1e-8
            
            polyhedron.add_ineq_constraint(g, constraint_value)
            
            endtime = time.time()
            print(f'End of iteration {k}. The whole iteration took {endtime - starttime}s.')
            
            if abs(UB - LB) <= epsilon:
                print(f"\nCONVERGED after {k} iterations!")
                break
                
        except Exception as e:
            print(f"Error in iteration {k}: {str(e)}")
            import traceback
            traceback.print_exc()
            break
            
        k += 1
    
    if best_lambdas is not None:
        return lambda x, y: f_tilde_cartesian(x, y, best_demands_locations, best_lambdas[0], best_v_tilde)
    else:
        print("No solution found.")
        return lambda x, y: 1.0 / (region.side_length ** 2)

def minimize_problem14_cartesian_debugged(demands, lambdas, t, region, use_torchquad=True):
    n = len(demands)
    v0_initial = 1.0
    v_initial = np.zeros(n)
    
    def objective(v):
        v0 = v[0]
        vi = v[1:]
        
        if use_torchquad:
            integral_val = compute_upper_bound_torchquad_debug(np.array([d.get_coordinates() for d in demands]), lambdas, v, region)
        else:
            integral_val, _ = compute_upper_bound_scipy(np.array([d.get_coordinates() for d in demands]), lambdas, v, region)
        
        return v0 * t + np.dot(vi, [d.dmd for d in demands]) - integral_val
        
    v_initial_guess = np.concatenate(([v0_initial], v_initial))
    
    result = optimize.minimize(objective, v_initial_guess, method='Nelder-Mead', options={'disp': False})
    
    return result.x, result.fun

def minimize_problem7_cartesian_debugged(lambdas, demands, t, region, use_torchquad=True):
    n = len(demands)
    v0_initial = 1.0
    v_initial = np.zeros(n)
    
    def objective(v):
        v0 = v[0]
        vi = v[1:]
        
        if use_torchquad:
             integral_val = compute_lower_bound_torchquad_debug(np.array([d.get_coordinates() for d in demands]), lambdas, v, region)
        else:
            integral_val, _ = compute_lower_bound_scipy(np.array([d.get_coordinates() for d in demands]), lambdas, v, region)
            
        return -(v0 * t + np.dot(vi, [d.dmd for d in demands]) - integral_val)

    v_initial_guess = np.concatenate(([v0_initial], v_initial))
    
    result = optimize.minimize(objective, v_initial_guess, method='Nelder-Mead', options={'disp': False})
    
    return result.x, -result.fun

def compute_upper_bound_torchquad_debug(demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(lambda X: integrand_upper_torchquad(X, demands_locations, lambdas, v), dim=2, N=1000, integration_domain=integration_domain, backend='torch')
    return result.item()

def compute_lower_bound_torchquad_debug(demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(lambda X: integrand_lower_torchquad(X, demands_locations, lambdas, v), dim=2, N=1000, integration_domain=integration_domain, backend='torch')
    return result.item()

def compute_g_integral_torchquad_debug(i, demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(lambda X: integrand_g_torchquad(X, i, demands_locations, lambdas, v), dim=2, N=1000, integration_domain=integration_domain, backend='torch')
    return result.item()

def get_subregion_index(x, y, demands_locations, lambdas):
    point = np.array([x, y])
    modified_distances = [np.linalg.norm(point - demands_locations[i]) - lambdas[i] for i in range(len(demands_locations))]
    return np.argmin(modified_distances)

def f_tilde_cartesian(x, y, demands_locations, lambdas, v_tilde):
    idx = get_subregion_index(x, y, demands_locations, lambdas)
    v0 = v_tilde[0]
    vi = v_tilde[idx + 1]
    distance = np.linalg.norm(np.array([x, y]) - demands_locations[idx])
    denominator = v0 * distance + vi
    return 1 / (denominator + 1e-9)

def integrand_lower_torchquad(X, demands_locations, lambdas, v_tilde):
    x, y = X[..., 0], X[..., 1]
    points = torch.stack([x, y], dim=-1)
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)
    modified_distances = torch.norm(points.unsqueeze(1) - demands_tensor, dim=2) - lambdas_tensor
    indices = torch.argmin(modified_distances, dim=1)
    selected_demands = demands_tensor[indices]
    v0 = torch.tensor(v_tilde[0], device=X.device)
    v_others = torch.from_numpy(v_tilde[1:]).to(X.device)
    selected_v = v_others[indices]
    distances = torch.norm(points - selected_demands, dim=1)
    denominators = v0 * distances + selected_v
    return 1 / (denominators + 1e-9)

def f_bar_cartesian(x, y, demands_locations, lambdas, v_bar):
    min_val = min_modified_norm(x, y, demands_locations, v_bar)
    return 1 / min_val if min_val > 0 else 0

def integrand_upper_torchquad(X, demands_locations, lambdas, v_bar):
    x, y = X[..., 0], X[..., 1]
    v0, v_others = v_bar[0], v_bar[1:]
    points = torch.stack([x, y], dim=-1)
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    v_others_tensor = torch.from_numpy(v_others).to(X.device)
    
    modified_norms = v0 * torch.norm(points.unsqueeze(1) - demands_tensor, dim=2) + v_others_tensor
    min_modified_norms, _ = torch.min(modified_norms, dim=1)
    
    return 1.0 / min_modified_norms

def integrand_g_torchquad(X, i, demands_locations, lambdas, v_bar):
    x, y = X[..., 0], X[..., 1]
    v0, v_others = v_bar[0], v_bar[1:]
    points = torch.stack([x, y], dim=-1)
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    v_others_tensor = torch.from_numpy(v_others).to(X.device)
    
    modified_norms = v0 * torch.norm(points.unsqueeze(1) - demands_tensor, dim=2) + v_others_tensor
    min_modified_norms, indices = torch.min(modified_norms, dim=1)
    
    # Create a mask where the ith component is the minimum
    mask = (indices == i).float()
    
    return mask / min_modified_norms

def min_modified_norm(x, y, demands_locations, v):
    v0 = v[0]
    vi = v[1:]
    min_val = np.inf
    for i in range(len(demands_locations)):
        dist = np.linalg.norm(np.array([x, y]) - demands_locations[i])
        val = v0 * dist + vi[i]
        if val < min_val:
            min_val = val
    return min_val 