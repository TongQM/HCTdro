import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy import optimize, integrate, linalg
from classes_cartesian_fixed import SquareRegion, Coordinate, DemandsGenerator, Polyhedron, Demand
import torch
import torchquad
import warnings
warnings.filterwarnings('ignore')


def find_worst_tsp_density_precise_fixed(region: SquareRegion, demands, t: float = 1, epsilon: float = 0.1, tol: float = 1e-4, use_torchquad: bool = True, max_iterations: int = 5):
    '''
    Fixed precise method: Analytic center cutting plane method with correct lower bound calculation.
    '''
    n = demands.shape[0]
    UB, LB = np.inf, -np.inf
    lambdas_bar = np.zeros(n)
    bound_value = min(region.side_length, 0.5)
    
    A = np.eye(n)
    b = bound_value * np.ones(n)
    B = np.ones((1, n))
    c = np.array([0.0])
    
    polyhedron = Polyhedron(A, b, B, c, n)
    k = 1
    best_lambdas = None
    best_v_tilde = None
    best_demands_locations = None
    
    while (abs(UB - LB) > epsilon) and (k <= max_iterations):
        print(f'\nIteration {k}:')
        starttime = time.time()
        
        try:
            lambdas_bar, _ = polyhedron.find_analytic_center_with_phase1(lambdas_bar)
            demands_locations = np.array([d.get_coordinates() for d in demands])
            
            # --- Corrected Upper Bound Calculation ---
            v_bar, _ = minimize_problem14_cartesian_fixed(demands, lambdas_bar, t, region)
            UB = compute_UB_from_v_bar(demands_locations, lambdas_bar, v_bar, region)
            
            # --- Corrected Lower Bound Calculation ---
            v_tilde, _ = minimize_problem7_cartesian_fixed(lambdas_bar, demands, t, region)
            LB = compute_LB_from_v_tilde(demands_locations, lambdas_bar, v_tilde, region)
            
            print(f"  UB={UB:.4f}, LB={LB:.4f}, gap={abs(UB-LB):.4f}")

            if best_lambdas is None or LB > best_lambdas[1]:
                best_lambdas = (lambdas_bar.copy(), LB)
                best_v_tilde = v_tilde.copy()
                best_demands_locations = demands_locations.copy()
            
            g = np.zeros(len(demands))
            for i in range(len(demands)):
                # --- Corrected g-vector Calculation ---
                g[i] = compute_g_integral_torchquad_fixed(i, demands_locations, lambdas_bar, v_bar, region)
            
            if np.linalg.norm(g) > 1e-6:
                g = g / np.linalg.norm(g)
            
            polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar + 1e-8)
            
            if abs(UB - LB) <= epsilon:
                print(f"Converged after {k} iterations.")
                break
                
        except Exception as e:
            print(f"Error in iteration {k}: {e}")
            break
            
        k += 1
    
    if best_lambdas is not None:
        # The f_tilde function constructed from the optimal v_tilde should already integrate to 1.
        # The final returned function is the unnormalized f_tilde.
        print(f"\nAlgorithm finished. Returning density function based on best LB={best_lambdas[1]:.6f}")
        return lambda x, y: f_tilde_cartesian(x, y, best_demands_locations, best_lambdas[0], best_v_tilde)
    else:
        # Fallback to uniform distribution if no solution was found
        print("\nAlgorithm failed to find a solution. Returning uniform distribution.")
        return lambda x, y: 1.0 / (region.side_length ** 2)

def find_min_dist_lambda(demands_locations, lambdas, region):
    """
    Finds the minimum value of min_i{||x - x_i|| - lambda_i} over the whole region R.
    This is used to handle the semi-infinite constraint in Problem 14.
    """
    def objective_func(x):
        point = np.array(x)
        modified_distances = [np.linalg.norm(point - demands_locations[i]) - lambdas[i] for i in range(len(demands_locations))]
        return min(modified_distances)

    bounds = [(region.x_min, region.x_max), (region.y_min, region.y_max)]
    initial_guesses = np.vstack([demands_locations, [[0,0], [region.x_min, region.y_min], [region.x_max, region.y_max]]])
    
    min_val = np.inf
    for guess in initial_guesses:
        res = optimize.minimize(objective_func, guess, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            
    return min_val

def minimize_problem14_cartesian_fixed(demands, lambdas, t, region):
    """
    Solves Problem 14 to find the upper bound.
    Optimizes over v0 and v1.
    """
    demands_locations = np.array([d.get_coordinates() for d in demands])

    # For the constraint v0*h(x)+v1 >= 0, we find the minimum of h(x) first.
    min_dist_lambda = find_min_dist_lambda(demands_locations, lambdas, region)

    def objective(v):
        # v has shape (2,) -> [v0, v1]
        v0, v1 = v[0], v[1]
        
        integral = compute_upper_bound_integral_fixed(demands_locations, lambdas, v, region)
        
        return integral + v0 * t + v1

    # v0 > 0, v1 >= 0
    # Set a small positive lower bound for v0 to ensure it's non-trivial
    bounds = [(1e-6, None), (0, None)]
    
    # Constraint: v0 * min_dist_lambda + v1 >= 0
    constraints = [{'type': 'ineq', 'fun': lambda v: v[0] * min_dist_lambda + v[1]}]
    
    v_initial_guess = np.ones(2)
    
    res = optimize.minimize(objective, v_initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False})
    
    # res.fun is the optimal UB, res.x is the optimal [v0, v1]
    return res.x, res.fun

def minimize_problem7_cartesian_fixed(lambdas, demands, t, region):
    n = len(demands)
    demands_locations = np.array([d.get_coordinates() for d in demands])

    def objective(v):
        # v has shape (n+1,)
        v0 = v[0]
        vi = v[1:]
        integral_val = compute_lower_bound_torchquad_fixed(demands_locations, lambdas, v, region)
        # Objective from paper: (1/4) * integral + v0*t + (1/n) * sum(vi)
        obj = 0.25 * integral_val + v0 * t + np.sum(vi) / n
        return obj

    # Constraints from paper: v_i >= 0 for all i=0..n
    # This makes the other constraints v0*||x-xi||+vi >= 0 redundant.
    bounds = [(0, None)] * (n + 1)

    # Initial guess for v
    v_initial_guess = np.ones(n + 1)

    res = optimize.minimize(objective, v_initial_guess, method='SLSQP', bounds=bounds, options={'disp': False})

    # res.fun is the optimal value (LB), res.x is the optimal v_tilde
    return res.x, res.fun

def compute_upper_bound_integral_fixed(demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    v_tensor = torch.from_numpy(v).double()
    return simpson.integrate(
        lambda X: integrand_upper_torchquad_fixed(X, demands_locations, lambdas, v_tensor),
        dim=2, N=1000, integration_domain=[[region.x_min, region.x_max], [region.y_min, region.y_max]]
    ).item()

def compute_lower_bound_torchquad_fixed(demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    return simpson.integrate(lambda X: integrand_lower_torchquad(X, demands_locations, lambdas, v), dim=2, N=1000, integration_domain=[[region.x_min, region.x_max], [region.y_min, region.y_max]]).item()

def compute_g_integral_torchquad_fixed(i, demands_locations, lambdas, v_bar, region):
    simpson = torchquad.Simpson()
    v_bar_tensor = torch.from_numpy(v_bar).double()
    # According to the paper, g_i = - integral over R_i of f_bar
    integral_val = simpson.integrate(
        lambda X: integrand_g_torchquad_fixed(X, i, demands_locations, lambdas, v_bar_tensor), 
        dim=2, N=1000, integration_domain=[[region.x_min, region.x_max], [region.y_min, region.y_max]]
    ).item()
    return -integral_val

def get_subregion_index(X, demands_locations, lambdas_tensor):
    # Unsqueeze demands_locations to enable broadcasting against the points tensor X
    modified_distances = torch.norm(X.unsqueeze(1) - demands_locations.unsqueeze(0), dim=2) - lambdas_tensor
    return torch.argmin(modified_distances, dim=1)

def f_tilde_cartesian(x, y, demands_locations, lambdas, v_tilde):
    """
    Computes the value of the piecewise density function f_tilde at a batch of points.
    Accepts tensor inputs for x and y from torchquad.
    """
    # Ensure x and y are tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # Combine x and y into a single points tensor
    points = torch.stack([x, y], dim=-1).double()

    demands_tensor = torch.from_numpy(demands_locations).to(points.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(points.device)
    v_tilde_tensor = torch.from_numpy(v_tilde).to(points.device).double()

    # Determine subregion index for each point in the batch
    indices = get_subregion_index(points, demands_tensor, lambdas_tensor)

    # Gather corresponding demand locations and v_tilde values
    selected_demands = demands_tensor[indices]
    selected_v = v_tilde_tensor[1:][indices]

    # Calculate distances and denominators for the batch
    distances = torch.norm(points - selected_demands, dim=1)
    denominators = v_tilde_tensor[0] * distances + selected_v
    
    # Definition from paper: f_tilde = (1/4) * (v0*||x-xi|| + vi)^-2
    return 0.25 / ((denominators + 1e-9)**2)

def integrand_lower_torchquad(X, demands_locations, lambdas, v_tilde):
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)
    v_tilde_tensor = torch.from_numpy(v_tilde).to(X.device).double()
    
    indices = get_subregion_index(X, demands_tensor, lambdas_tensor)
    
    selected_demands = demands_tensor[indices]
    selected_v = v_tilde_tensor[1:][indices]
    
    distances = torch.norm(X - selected_demands, dim=1)
    denominators = v_tilde_tensor[0] * distances + selected_v
    return 1 / (denominators + 1e-9)

def integrand_upper_torchquad_fixed(X, demands_locations, lambdas, v):
    # v is a 2-element tensor [v0, v1]
    v0, v1 = v[0], v[1]
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)

    # h(x,λ) = min_j{||x - x_j|| - λ_j}
    modified_distances = torch.norm(X.unsqueeze(1) - demands_tensor, dim=2) - lambdas_tensor
    min_modified_distances, _ = torch.min(modified_distances, dim=1)

    denominator = 4 * (v0 * min_modified_distances + v1)
    
    return 1.0 / (denominator + 1e-9)

def integrand_g_torchquad_fixed(X, i, demands_locations, lambdas, v_bar):
    """
    This is the integrand for calculating g_i.
    It computes f_bar(x) and applies a mask to only consider the subregion R_i.
    The integral of this function gives ∫_{R_i} f_bar(x) dA.
    """
    # v_bar is a 2-element tensor [v0, v1]
    v0, v1 = v_bar[0], v_bar[1]
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)

    # h(x,λ) = min_j{||x - x_j|| - λ_j}
    modified_distances = torch.norm(X.unsqueeze(1) - demands_tensor, dim=2) - lambdas_tensor
    min_modified_distances, indices = torch.min(modified_distances, dim=1)

    # f_bar(x) = 1 / (4 * (v0 * h(x,λ) + v1))
    denominator = 4 * (v0 * min_modified_distances + v1)
    f_bar_values = 1.0 / (denominator + 1e-9)
    
    # Mask for R_i (where i is the minimizing index for h(x,λ))
    mask = (indices == i).float()
    
    # The integrand for g_i is f_bar(x) restricted to the subregion R_i
    return f_bar_values * mask 

def compute_UB_from_v_bar(demands_locations, lambdas, v_bar, region):
    """Calculates UB = integral(sqrt(f_bar))"""
    simpson = torchquad.Simpson()
    v_bar_tensor = torch.from_numpy(v_bar).double()
    return simpson.integrate(
        lambda X: integrand_sqrt_f_bar(X, demands_locations, lambdas, v_bar_tensor),
        dim=2, N=1000, integration_domain=[[region.x_min, region.x_max], [region.y_min, region.y_max]]
    ).item()

def compute_LB_from_v_tilde(demands_locations, lambdas, v_tilde, region):
    """Calculates LB = integral(sqrt(f_tilde))"""
    simpson = torchquad.Simpson()
    v_tilde_tensor = torch.from_numpy(v_tilde).double()
    return simpson.integrate(
        lambda X: integrand_sqrt_f_tilde(X, demands_locations, lambdas, v_tilde_tensor),
        dim=2, N=1000, integration_domain=[[region.x_min, region.x_max], [region.y_min, region.y_max]]
    ).item()

def integrand_sqrt_f_bar(X, demands_locations, lambdas, v_bar):
    # sqrt(f_bar) = (1/2) * (v0*h(x,λ) + v1)^-1
    v0, v1 = v_bar[0], v_bar[1]
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)
    modified_distances = torch.norm(X.unsqueeze(1) - demands_tensor, dim=2) - lambdas_tensor
    min_modified_distances, _ = torch.min(modified_distances, dim=1)
    denominator = 2 * (v0 * min_modified_distances + v1)
    return 1.0 / (denominator + 1e-9)

def integrand_sqrt_f_tilde(X, demands_locations, lambdas, v_tilde):
    # sqrt(f_tilde) = (1/2) * (v0*||x-xi|| + vi)^-1
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)
    v_tilde_tensor = v_tilde.to(X.device)
    indices = get_subregion_index(X, demands_tensor, lambdas_tensor)
    selected_demands = demands_tensor[indices]
    selected_v = v_tilde_tensor[1:][indices]
    distances = torch.norm(X - selected_demands, dim=1)
    denominator = 2 * (v_tilde_tensor[0] * distances + selected_v)
    return 1 / (denominator + 1e-9) 