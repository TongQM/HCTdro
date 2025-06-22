import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy import optimize, integrate, linalg
from classes_cartesian import SquareRegion, Coordinate, DemandsGenerator, Polyhedron, Demand
import torch
import torchquad


def find_worst_tsp_density_precise(region: SquareRegion, demands, t: float = 1, epsilon: float = 0.1, tol: float = 1e-4, use_torchquad: bool = True):
    '''
    Precise method: Analytic center cutting plane method for finding worst-case TSP density
    Adapted for Cartesian coordinates on a square region
    
    This algorithm finds the worst TSP density using the analytic center cutting plane method.
    
    Input: A square region containing a set of distinct points x1, x2,..., xn, which are 
    interpreted as an empirical distribution f_hat, a distance parameter t, and a tolerance epsilon.

    Output: An epsilon-approximation of the distribution f* that maximizes int_R sqrt(f(x)) dA 
    subject to the constraint that D(f_hat, f) <= t.
    '''

    n = demands.shape[0]
    UB, LB = np.inf, -np.inf
    lambdas_bar = np.zeros(n)
    # Initialize polyhedron with bounds on lambda variables
    polyhedron = Polyhedron(np.eye(n), region.side_length * np.ones(n), np.ones((1, n)), 0, n)
    k = 1
    
    while (abs(UB - LB) > epsilon):
        print(f'Iteration {k} begins: \n')
        starttime = time.time()
        
        # Find analytic center
        lambdas_bar, lambdas_bar_func_val = polyhedron.find_analytic_center(lambdas_bar)
        time1 = time.time()
        print(f'Find analytic center: Lambdas_bar is {lambdas_bar}, with value {lambdas_bar_func_val}, took {time1 - starttime}s.')

        demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])

        # Build an upper bounding f_bar for the original problem
        v_bar, problem14_func_val = minimize_problem14_cartesian(demands, lambdas_bar, t, region, use_torchquad)
        if use_torchquad:
            UB = compute_upper_bound_torchquad(demands_locations, lambdas_bar, v_bar, region)
        else:
            UB, UB_error = compute_upper_bound_scipy(demands_locations, lambdas_bar, v_bar, region, tol)
        time2 = time.time()
        print(f'Find upper bound: Upper bound is {UB}, took {time2 - time1}s.')

        # Build a lower bounding f_tilde that is feasible for the original problem
        v_tilde, problem7_func_val = minimize_problem7_cartesian(lambdas_bar, demands, t, region, use_torchquad)
        if use_torchquad:
            LB = compute_lower_bound_torchquad(demands_locations, lambdas_bar, v_tilde, region)
        else:
            LB, LB_error = compute_lower_bound_scipy(demands_locations, lambdas_bar, v_tilde, region, tol)
        time3 = time.time()
        print(f'Find lower bound: Lower bound is {LB}, took {time3 - time2}s.\n')

        # Update g vector for cutting plane
        g = np.zeros(len(demands))
        for i in range(len(demands)):
            if use_torchquad:
                g[i] = compute_g_integral_torchquad(i, demands_locations, lambdas_bar, v_bar, region)
            else:
                g[i], g_error = compute_g_integral_scipy(i, demands_locations, lambdas_bar, v_bar, region, tol)

        # Update polyhedron Lambda to get next analytic center
        polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)
        time4 = time.time()
        print(f'It took {time4 - time3}s to get vector g.\n')

        endtime = time.time()
        print(f'End of iteration {k}.\n  The whole iteration took {endtime - starttime}s.\n')
        k += 1

    # Return the worst-case density function
    return lambda x, y: f_tilde_cartesian(x, y, demands_locations, lambdas_bar, v_tilde)


@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


@nb.jit(nopython=True)
def min_modified_norm(x_coord, demands_locations, lambdas):
    n = demands_locations.shape[0]
    norms = np.array([norm_func(x_coord, demands_locations[i]) - lambdas[i] for i in range(n)])
    return np.min(norms)


@nb.jit(nopython=True)
def modified_norm(x_coord, i, demands_locations, lambdas):
    return norm_func(x_coord, demands_locations[i]) - lambdas[i]


@nb.jit(nopython=True)
def region_indicator(i, x_coord, lambdas, demands_locations):
    i_modified_norm = modified_norm(x_coord, i, demands_locations, lambdas)
    for j in range(len(lambdas)):
        if i_modified_norm > modified_norm(x_coord, j, demands_locations, lambdas):
            return 0
    return 1


@nb.jit(nopython=True)
def categorize_x(x_coord, demands_locations, lambdas, v):
    modified_norms = np.array([modified_norm(x_coord, i, demands_locations, lambdas) for i in range(demands_locations.shape[0])])
    i = np.argmin(modified_norms)
    return demands_locations[i], v[i+1]


def f_bar_cartesian(x, y, demands_locations, lambdas_bar, v_bar):
    x_coord = np.array([x, y])
    return 1/4 * pow((v_bar[0]*min_modified_norm(x_coord, demands_locations, lambdas_bar) + v_bar[1]), -2)


def get_subregion_index(x, y, demands_locations, lambdas):
    """
    Determines the index of the subregion R_i^* that the point (x, y) belongs to.
    The subregion is defined by the lambdas.
    """
    point = np.array([x, y])
    modified_distances = [np.linalg.norm(point - demands_locations[i]) - lambdas[i] for i in range(len(demands_locations))]
    return np.argmin(modified_distances)


def f_tilde_cartesian(x, y, demands_locations, lambdas, v_tilde):
    """
    Computes the value of the piecewise density function f_tilde at a point (x, y).
    The partition is defined by lambdas, while the function value on each piece
    is defined by v_tilde.
    """
    idx = get_subregion_index(x, y, demands_locations, lambdas)
    
    # The density is 1 / (v_tilde[0] * ||x - x_i|| + v_tilde[i+1])
    # where i is the index of the subregion.
    v0 = v_tilde[0]
    vi = v_tilde[idx + 1]
    
    distance = np.linalg.norm(np.array([x, y]) - demands_locations[idx])
    
    denominator = v0 * distance + vi
    
    # Add a small epsilon to avoid division by zero if denominator is too small
    return 1 / (denominator + 1e-9)


# Torchquad implementations
def integrand_upper_torchquad(X, demands_locations, lambdas, v):
    '''
    X is a n-by-2 matrix, the first column is x, the second column is y.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    
    norms = torch.cdist(X, demands_locations, p=2)
    modified_norms, _ = torch.min(norms - lambdas, dim=1)
    raw_intgrd = 1/(4*(v0*modified_norms + v1))
    
    return raw_intgrd


def integrand_lower_torchquad(X, demands_locations, lambdas, v_tilde):
    """
    Integrand for the lower bound calculation using torchquad.
    This function correctly uses lambdas to define the partition.
    """
    x, y = X[..., 0], X[..., 1]
    
    # Determine the subregion index for a batch of points
    points = torch.stack([x, y], dim=-1)
    demands_tensor = torch.from_numpy(demands_locations).to(X.device)
    lambdas_tensor = torch.from_numpy(lambdas).to(X.device)
    
    # Broadcasting to compute all modified distances at once
    # Shape: (batch_size, num_demands)
    modified_distances = torch.norm(points.unsqueeze(1) - demands_tensor, dim=2) - lambdas_tensor
    
    # Find the index of the minimum for each point in the batch
    # Shape: (batch_size,)
    indices = torch.argmin(modified_distances, dim=1)
    
    # Gather the corresponding demand locations and v_tilde values
    selected_demands = demands_tensor[indices] # Shape: (batch_size, 2)
    
    v0 = torch.tensor(v_tilde[0], device=X.device)
    v_others = torch.from_numpy(v_tilde[1:]).to(X.device)
    selected_v = v_others[indices] # Shape: (batch_size,)
    
    # Calculate the denominator for each point
    distances = torch.norm(points - selected_demands, dim=1)
    denominators = v0 * distances + selected_v
    
    return 1 / (denominators + 1e-9)


def integrand_g_torchquad(X, i, demands_locations, lambdas, v):
    '''
    X is a n-by-2 matrix, the first column is x, the second column is y.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    
    norms = torch.cdist(X, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    
    # Indicator function: 1 if i is the argmin, 0 otherwise
    indicator = (modified_norms_indices == i).float()
    
    raw_intgrd = 1/(4*(v0*modified_norms + v1))
    return indicator * raw_intgrd


def compute_upper_bound_torchquad(demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(
        lambda X: integrand_upper_torchquad(X, demands_locations, lambdas, v), 
        dim=2, N=100000, integration_domain=integration_domain, backend='torch'
    )
    return result.item()


def compute_lower_bound_torchquad(demands_locations, lambdas, v_tilde, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(
        lambda X: integrand_lower_torchquad(X, demands_locations, lambdas, v_tilde), 
        dim=2, N=100000, integration_domain=integration_domain, backend='torch'
    )
    return result.item()


def compute_g_integral_torchquad(i, demands_locations, lambdas, v, region):
    simpson = torchquad.Simpson()
    integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
    result = simpson.integrate(
        lambda X: integrand_g_torchquad(X, i, demands_locations, lambdas, v), 
        dim=2, N=100000, integration_domain=integration_domain, backend='torch'
    )
    return result.item()


# Scipy implementations
def integrand_upper_scipy(x, y, demands_locations, lambdas, v):
    x_coord = np.array([x, y])
    raw_intgrd = 1/(4*(v[0]*min_modified_norm(x_coord, demands_locations, lambdas) + v[1]))
    return raw_intgrd


def integrand_lower_scipy(x, y, demands_locations, lambdas, v):
    x_coord = np.array([x, y])
    xi, vi = categorize_x(x_coord, demands_locations, lambdas, v)
    raw_intgrd = 1 / (v[0]*norm_func(x_coord, xi) + vi)
    return raw_intgrd


def integrand_g_scipy(x, y, i, demands_locations, lambdas, v):
    x_coord = np.array([x, y])
    if region_indicator(i, x_coord, lambdas, demands_locations) == 0:
        return 0
    return f_bar_cartesian(x, y, demands_locations, lambdas, v)


def compute_upper_bound_scipy(demands_locations, lambdas, v, region, tol):
    return integrate.dblquad(
        lambda y, x: integrand_upper_scipy(x, y, demands_locations, lambdas, v),
        region.x_min, region.x_max,
        lambda _: region.y_min, lambda _: region.y_max,
        epsabs=tol
    )


def compute_lower_bound_scipy(demands_locations, lambdas, v, region, tol):
    return integrate.dblquad(
        lambda y, x: integrand_lower_scipy(x, y, demands_locations, lambdas, v),
        region.x_min, region.x_max,
        lambda _: region.y_min, lambda _: region.y_max,
        epsabs=tol
    )


def compute_g_integral_scipy(i, demands_locations, lambdas, v, region, tol):
    return integrate.dblquad(
        lambda y, x: integrand_g_scipy(x, y, i, demands_locations, lambdas, v),
        region.x_min, region.x_max,
        lambda _: region.y_min, lambda _: region.y_max,
        epsabs=tol
    )


# Problem 14 (upper bound) implementation
def objective_function_problem14(v, demands_locations, lambdas, t, region, use_torchquad):
    if use_torchquad:
        simpson = torchquad.Simpson()
        integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
        area = simpson.integrate(
            lambda X: integrand_upper_torchquad(X, demands_locations, lambdas, v), 
            dim=2, N=100000, integration_domain=integration_domain, backend='torch'
        ).item()
    else:
        area, _ = integrate.dblquad(
            lambda y, x: integrand_upper_scipy(x, y, demands_locations, lambdas, v),
            region.x_min, region.x_max,
            lambda _: region.y_min, lambda _: region.y_max,
            epsabs=1e-4
        )
    return area + v[0]*t + v[1]


def minimize_problem14_cartesian(demands, lambdas, t, region, use_torchquad=True):
    demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])
    
    # Constraint: v[0] * min_modified_norm + v[1] >= 0 for all x in region
    # We can use a simple constraint based on the minimum possible distance
    min_distance = 0  # This could be refined based on the region geometry
    constraint_coeff = np.array([min_distance, 1])
    constraints_dict = {'type': 'ineq', 'fun': lambda v: constraint_coeff @ v, 'jac': lambda _: constraint_coeff}
    
    bounds = optimize.Bounds(0.0001, np.inf)
    result = optimize.minimize(
        objective_function_problem14, 
        x0=np.array([0.0001, 1]), 
        args=(demands_locations, lambdas, t, region, use_torchquad), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints_dict,
        options={'ftol': 1e-4, 'disp': False}
    )
    return result.x, result.fun


# Problem 7 (lower bound) implementation
def objective_function_problem7(v, demands_locations, lambdas, t, region, use_torchquad):
    if use_torchquad:
        simpson = torchquad.Simpson()
        integration_domain = [[region.x_min, region.x_max], [region.y_min, region.y_max]]
        area = simpson.integrate(
            lambda X: integrand_lower_torchquad(X, demands_locations, lambdas, v), 
            dim=2, N=100000, integration_domain=integration_domain, backend='torch'
        ).item()
    else:
        area, _ = integrate.dblquad(
            lambda y, x: integrand_lower_scipy(x, y, demands_locations, lambdas, v),
            region.x_min, region.x_max,
            lambda _: region.y_min, lambda _: region.y_max,
            epsabs=1e-4
        )
    return 1/4*area + v[0]*t + np.mean(v[1:])


def minimize_problem7_cartesian(lambdas, demands, t, region, use_torchquad=True):
    demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])
    bounds = optimize.Bounds(0, np.inf)
    result = optimize.minimize(
        objective_function_problem7, 
        x0=np.append(1e-6, np.ones(demands.shape[0])), 
        args=(demands_locations, lambdas, t, region, use_torchquad), 
        method='SLSQP', 
        bounds=bounds,
        options={'ftol': 1e-4, 'disp': False}
    )
    return result.x, result.fun 