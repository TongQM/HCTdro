from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from scipy import optimize, integrate, linalg
from torch import FloatStorage
from classes import Region, Coordinate, Demands_generator, Demand


# n = 2
# np.random.seed(11)
# region = Region(2)
# depot = Coordinate(2, 0.3)
# generator = Demands_generator(region, n)
# demands = generator.generate()
# lambdas_temporary = np.zeros(n)
# t_temporary = 1

#error_model="numpy" -> Don't check for division by zero
# @nb.njit(error_model="numpy",fastmath=True)

@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


@nb.jit(nopython=True)
def modified_norm(x_cdnt: list[float], i: int, demands_locations: list[list[float]], lambdas: list[float]):
    return norm_func(x_cdnt, demands_locations[i]) - lambdas[i]

@nb.jit(nopython=True)
def integrand(r: float, theta: float, lambdas: list[float], v: list[float], demands_locations: list[list[float]]) -> float:
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    xi, vi = categorize_x(x_cdnt, demands_locations, lambdas, v)
    raw_intgrd = 1 / (v[0]*norm_func(x_cdnt, xi) + vi)
    return r*raw_intgrd

@nb.jit(nopython=True)
def region_indicator(i: int, x_cdnt: list[float], lambdas: list[float], demands_locations: list[list[float]]) -> Literal[0, 1]:
    i_modified_norm = modified_norm(x_cdnt, i, demands_locations, lambdas)
    for j in range(len(lambdas)):
        if i_modified_norm > modified_norm(x_cdnt, j, demands_locations, lambdas):
            return 0
    return 1

@nb.jit(nopython=True)
def categorize_x(x_cdnt: list[float], demands_locations: list[list[float]], lambdas: list[float], v: list[float]):
    modified_norms = np.array([modified_norm(x_cdnt, i, demands_locations, lambdas) for i in range(demands_locations.shape[0])])
    i = np.argmin(modified_norms)
    return demands_locations[i], v[i+1]


def objective_function(v: list[float], demands_locations: list[list[float]], lambdas: list[float], t: float, region_radius) -> float:
    sum_integral, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(lambdas, v, demands_locations), epsabs=1e-3)
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])


def minimize_problem7(lambdas: list[float], demands: list[Demand], t: float, region_radius) -> list[float]:
    # constraints = [optimize.NonlinearConstraint(lambda v: constraint_func(lambdas, demands, v, region_radius), 0, np.inf)]
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    bounds = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function, x0=np.append(np.zeros(1), np.ones(demands.size)), args=(demands_locations, lambdas, t, region_radius), method='SLSQP',  bounds=bounds)
    return result.x, result.fun



'''Pre-satisfied constraints when bounding'''
def constraint_objective(x_cdnt, region_radius, v, demands, lambdas):
    xi, vi = categorize_x(x_cdnt, demands, lambdas, v)
    return v[0]*linalg.norm(x_cdnt - xi.get_cdnt()) + vi


def constraint_func(lambdas, demands, v, region_radius):
    objective = lambda x_cdnt: constraint_objective(x_cdnt, region_radius, v, demands, lambdas)
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region_radius)
    result = optimize.minimize(objective, x0=np.zeros(2), method='SLSQP', constraints=x_in_R_constraint)
    return result.fun


# x, fun_val = minimize_problem7(lambdas_temporary, demands, t_temporary, region)