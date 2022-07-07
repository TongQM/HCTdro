import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator

# n = 2
# np.random.seed(11)
# region = Region(2)
# depot = Coordinate(2, 0.3)
# generator = Demands_generator(region, n)
# demands = generator.generate()
# lambdas_temporary = np.zeros(n)
# t_temporary = 1
# v0_temporary = 1
# v1_temporary = 1
# rs, thetas = [d.location.r for d in demands], [d.location.theta for d in demands]
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.scatter(thetas, rs)

#error_model="numpy" -> Don't check for division by zero
# @nb.njit(error_model="numpy",fastmath=True)

@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@nb.jit(nopython=True)
def min_modified_norm(x_cdnt, demands_locations, lambdas):
    n = demands_locations.size
    norms = np.array([norm_func(x_cdnt, demands_locations[i]) - lambdas[i] for i in range(n)])
    return np.min(norms)

@nb.jit(nopython=True)
def integrand(r: float, theta: float, v, demands_locations, lambdas):
    # Calculate a list of ||x-xi|| - lambda_i
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    raw_intgrd = 1/(4*(v[0]*min_modified_norm(x_cdnt, demands_locations, lambdas) + v[1]))
    return raw_intgrd*r    # r as Jacobian

# @nb.jit(nopython=True)
def objective_function(v, demands_locations, lambdas, t, region_radius):
    area, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(v, demands_locations, lambdas), epsabs=1e-3)
    return area + v[0]*t + v[1]


def constraint_coeff(demands_locations, lambdas, region_radius):
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region_radius)
    result = optimize.minimize(lambda x_cdnt: min_modified_norm(x_cdnt, demands_locations, lambdas), x0 = np.ones(2), method='SLSQP', constraints=x_in_R_constraint)
    return np.array([result.fun, 1]) # because the constraint below is v0*modified_min_norm + v1 >= 0


def minimize_problem14(demands, lambdas, t, region_radius):
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    constraints = [optimize.LinearConstraint(constraint_coeff(demands_locations, lambdas, region_radius), 0, np.inf)]
    bound = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function, x0=np.ones(2), args=(demands_locations, lambdas, t, region_radius),method='SLSQP', bounds=bound, constraints=constraints)
    return result.x, result.fun


# v, func_value = minimize_problem14(demands, lambdas_temporary, t_temporary, region)