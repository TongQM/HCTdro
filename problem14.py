import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator


tol = 1e-3
@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@nb.jit(nopython=True)
def min_modified_norm(x_cdnt, demands_locations, lambdas):
    n = demands_locations.shape[0]
    norms = np.array([norm_func(x_cdnt, demands_locations[i]) - lambdas[i] for i in range(n)])
    return np.min(norms)

@nb.njit
def integrand(r: float, theta: float, v, demands_locations, lambdas):
    # Calculate a list of ||x-xi|| - lambda_i
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    raw_intgrd = 1/(4*(v[0]*min_modified_norm(x_cdnt, demands_locations, lambdas) + v[1]))
    return raw_intgrd*r    # r as Jacobian

@nb.njit
def jac_integrand0(r: float, theta: float, v, demands_locations, lambdas):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    the_min_modified_norm = min_modified_norm(x_cdnt, demands_locations, lambdas)
    raw_intgrd = -4*the_min_modified_norm / pow(4*(v[0]*the_min_modified_norm + v[1]), 2)
    return raw_intgrd*r

@nb.njit
def jac_integrand1(r: float, theta: float, v, demands_locations, lambdas):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    the_min_modified_norm = min_modified_norm(x_cdnt, demands_locations, lambdas)
    raw_intgrd = -4 / pow(4*(v[0]*the_min_modified_norm + v[1]), 2)
    return raw_intgrd*r


# @nb.jit(nopython=True)
def objective_function(v, demands_locations, lambdas, t, region_radius):
    area, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(v, demands_locations, lambdas), epsabs=tol)
    return area + v[0]*t + v[1]

def objective_jac(v, demands_locations, lambdas, t, region_radius):
    area0, error0 = integrate.dblquad(jac_integrand0, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(v, demands_locations, lambdas), epsabs=tol)
    area1, error1 = integrate.dblquad(jac_integrand1, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(v, demands_locations, lambdas), epsabs=tol)
    return np.array([area0 + t, area1 + 1])

def constraint_and_jac(demands_locations, lambdas, region_radius):
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region_radius)
    result = optimize.minimize(lambda x_cdnt: min_modified_norm(x_cdnt, demands_locations, lambdas), x0 = np.ones(2), method='SLSQP', constraints=x_in_R_constraint)
    return np.array([result.fun, 1]), np.array([result.fun, 1]) # because the constraint below is v0*modified_min_norm + v1 >= 0


def minimize_problem14(demands, lambdas, t, region_radius):
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    constraint_coeff, constraint_jac = constraint_and_jac(demands_locations, lambdas, region_radius)
    constraints_dict = {'type': 'ineq', 'fun': lambda v: constraint_coeff @ v, 'jac': lambda _: constraint_jac}
    bound = optimize.Bounds(0.0001, np.inf)
    result = optimize.minimize(objective_function, x0=np.array([0, 1]), args=(demands_locations, lambdas, t, region_radius), jac=objective_jac, method='SLSQP', bounds=bound, constraints=constraints_dict)
    return result.x, result.fun


# v, func_value = minimize_problem14(demands, lambdas_temporary, t_temporary, region)