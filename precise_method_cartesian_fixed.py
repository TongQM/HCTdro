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


def find_worst_tsp_density_precise_fixed(region: SquareRegion, demands, t: float = 1, epsilon: float = 0.1, tol: float = 1e-4, use_torchquad: bool = True, max_iterations: int = 5, return_params: bool = False, return_history: bool = False):
    '''
    Fixed precise method: Analytic center cutting plane method with correct lower bound calculation.
    '''
    n = len(demands)
    UB, LB = np.inf, -np.inf
    lambdas_bar = np.zeros(n)
    # bound_value = min(region.side_length, 0.5)
    bound_value = region.side_length * np.sqrt(2)
    # For convergence history
    ub_hist, lb_hist, gap_hist = [], [], []
    
    A = np.eye(n)
    b = bound_value * np.ones(n)
    B = np.ones((1, n))
    c = np.array([0.0])
    
    polyhedron = Polyhedron(A, b, B, c, n)
    k = 1
    best_lambdas = None
    best_v_tilde = None
    
    # Compute demand locations once
    demands_locations = np.array([d.get_coordinates() for d in demands])
    
    stagnation_counter = 0
    stagnation_tol = 1e-6
    stagnation_window = 5
    last_ub, last_lb, last_gap = None, None, None
    
    while (abs(UB - LB) > epsilon) and (k <= max_iterations):
        print(f'\nIteration {k}:')
        starttime = time.time()
        
        try:
            # Try Newton-based analytic center first, fallback to Phase I + trust-constr
            try:
                lambdas_bar, _ = polyhedron.find_analytic_center_newton(lambdas_bar, max_iters=200, tol=1e-8, verbose=True)
            except Exception as _e:
                print(f"  [Analytic Center] Newton method failed: {_e}. Falling back to Phase I + trust-constr")
                lambdas_bar, _ = polyhedron.find_analytic_center_with_phase1(lambdas_bar)
            print(f"  ||lambda|| = {np.linalg.norm(lambdas_bar):.6f}")
            print(f"  sum(lambda) = {np.sum(lambdas_bar):.6e} (should be zero)")
            # demands_locations is fixed, do not update
            
            # --- Corrected Upper Bound Calculation ---
            v_bar, _ = minimize_problem14_cartesian_fixed(demands, lambdas_bar, t, region)
            UB = compute_UB_from_v_bar(demands_locations, lambdas_bar, v_bar, region)
            
            # --- Corrected Lower Bound Calculation ---
            v_tilde, _ = minimize_problem7_cartesian_fixed(lambdas_bar, demands, t, region)
            LB = compute_LB_from_v_tilde(demands_locations, lambdas_bar, v_tilde, region)
            
            print(f"  UB={UB:.4f}, LB={LB:.4f}, gap={abs(UB-LB):.4f}")
            ub_hist.append(UB)
            lb_hist.append(LB)
            gap_hist.append(abs(UB-LB))

            # Stagnation detection
            if last_ub is not None and last_lb is not None and last_gap is not None:
                if (abs(UB - last_ub) < stagnation_tol and
                    abs(LB - last_lb) < stagnation_tol and
                    abs(abs(UB-LB) - last_gap) < stagnation_tol):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                if stagnation_counter >= stagnation_window:
                    print(f"Stagnation detected: UB/LB/gap unchanged for {stagnation_window} iterations. Stopping.")
                    break
            last_ub, last_lb, last_gap = UB, LB, abs(UB-LB)

            if best_lambdas is None or LB > best_lambdas[1]:
                best_lambdas = (lambdas_bar.copy(), LB)
                best_v_tilde = v_tilde.copy()
            
            g = np.zeros(n)
            for i in range(n):
                g[i] = compute_g_integral_torchquad_fixed(i, demands_locations, lambdas_bar, v_bar, region)
            
            if np.linalg.norm(g) > 1e-6:
                g = g / np.linalg.norm(g)
            
            # Add the cut with the correct sign: -g <= -g.T @ lambdas_bar enforces g^T lambda >= g^T lambdas_bar
            polyhedron.add_ineq_constraint(-g, -g.T @ lambdas_bar)
            
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
        density_func = lambda x, y: f_tilde_cartesian(x, y, demands_locations, best_lambdas[0], best_v_tilde)
        if return_history:
            history = {'UB': ub_hist, 'LB': lb_hist, 'gap': gap_hist}
        if return_params and return_history:
            return density_func, demands_locations, best_lambdas[0], best_v_tilde, history
        elif return_params:
            return density_func, demands_locations, best_lambdas[0], best_v_tilde
        elif return_history:
            return density_func, history
        else:
            return density_func
    else:
        # Fallback to uniform distribution if no solution was found
        print("\nAlgorithm failed to find a solution. Returning uniform distribution.")
        density_func = lambda x, y: 1.0 / (region.side_length ** 2)
        if return_history:
            history = {'UB': ub_hist, 'LB': lb_hist, 'gap': gap_hist}
        if return_params and return_history:
            return density_func, demands_locations, None, None, history
        elif return_params:
            return density_func, demands_locations, None, None
        elif return_history:
            return density_func, history
        else:
            return density_func

def find_min_dist_lambda(demands_locations, lambdas, region):
    """
    Finds the minimum value of min_i{||x - x_i|| - lambda_i} over the whole region R.
    
    Mathematical insight: Since ||x - x_i|| >= 0 for all x, the minimum value of
    ||x - x_i|| - lambda_i is achieved when ||x - x_i|| = 0 (i.e., x = x_i).
    Therefore: min_{x∈R} min_i{||x - x_i|| - lambda_i} = min_i{-lambda_i} = -max(lambda).
    """
    return -np.max(lambdas)

def minimize_problem14_cartesian_fixed(demands, lambdas, t, region):
    """
    Solves Problem 14 to find the upper bound.
    Optimizes over v0 and v1.
    """
    demands_locations = np.array([d.get_coordinates() for d in demands])

    # For the constraint v0*h(x)+v1 >= 0, we find the minimum of h(x) first.
    min_dist_lambda = find_min_dist_lambda(demands_locations, lambdas, region)
    # Original formulation uses >= 0, not >= delta

    def objective(v):
        v0, v1 = v[0], v[1]
        integral = compute_upper_bound_integral_fixed(demands_locations, lambdas, v, region)
        if integral > 1e6:
            print(f"[Warning] Integral in UB objective is huge ({integral:.2e}). Returning penalty.")
            return 1e8
        return integral + v0 * t + v1

    # Original formulation: v0, v1 >= 0 (no artificial lower bound)
    bounds = [(0.0, None), (0.0, None)]
    # Constraint: v0 * min_dist_lambda + v1 >= 0 (original formulation)
    constraints = [{'type': 'ineq', 'fun': lambda v: v[0] * min_dist_lambda + v[1]}]
    # Initial guess: always feasible for original constraint
    v0_init = 1.0
    v1_init = max(1.0, -v0_init * min_dist_lambda + 0.1)  # Small margin above 0
    v_initial_guess = np.array([v0_init, v1_init])
    res = optimize.minimize(objective, v_initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False})
    # Warnings for extreme v0, v1
    if res.x[0] < 1e-6 or res.x[1] < 1e-6 or res.x[0] > 1e3 or res.x[1] > 1e3:
        print(f"[Warning] v0 or v1 is extreme: v0={res.x[0]:.2e}, v1={res.x[1]:.2e}")
    # Warn if the constraint is nearly violated (numerical instability)
    if res.x[0] * min_dist_lambda + res.x[1] < -1e-6:
        print(f"[Warning] v0*min_dist_lambda + v1 = {res.x[0] * min_dist_lambda + res.x[1]:.2e} violates constraint.")
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
    Computes the value of the piecewise density function f_tilde at a batch of points or a single point.
    Accepts tensor inputs for x and y from torchquad, or scalars for single-point evaluation.
    """
    # Ensure x and y are tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # If x and y are scalars (0-dim), convert to 1-dim
    if x.dim() == 0 and y.dim() == 0:
        points = torch.stack([x, y], dim=0).double().unsqueeze(0)  # shape (1, 2)
    elif x.dim() == 1 and y.dim() == 1 and x.shape == y.shape:
        points = torch.stack([x, y], dim=-1).double()  # shape (N, 2)
    else:
        raise ValueError("x and y must be both scalars or both 1D tensors of the same shape.")

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
    result = 0.25 / ((denominators + 1e-9)**2)
    # If input was a single point, return a scalar
    if result.numel() == 1:
        return result.item()
    return result

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