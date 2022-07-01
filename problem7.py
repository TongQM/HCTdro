from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator, Demand

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



def modified_norm(x_cdnt: list[float], i: int, demands: list[Demand], lambdas: list[float]) -> float:
    return linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas[i]


def integrand(r: float, theta: float, lambdas: list[float], v: list[float], demands: list[Demand]) -> float:
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    raw_intgrd = np.sum([region_indicator(i, x_cdnt, lambdas, demands) / (v[0]*linalg.norm(x_cdnt - demands[i].get_cdnt()) + v[i+1]) for i in range(len(demands))])
    return r*raw_intgrd


def objective_function(demands: list[Demand], lambdas: list[float], t: float, v: list[float], region: Region) -> float:
    sum_integral, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(lambdas, v, demands), epsabs=1e-3)
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])


def region_indicator(i: int, x_cdnt: list[float], lambdas: list[float], demands: list[Demand]) -> Literal[0, 1]:
    for j in range(len(lambdas)):
        if modified_norm(x_cdnt, i, demands, lambdas) > modified_norm(x_cdnt, j, demands, lambdas):
            return 0
    return 1

def categorize_x(x_cdnt: list[float], demands: list[Demand], lambdas: list[float], v: list[float]):
    for i in range(len(demands)):
        if region_indicator(i, x_cdnt, lambdas, demands): return demands[i], v[i+1]
    assert('x not in the region!')


def constraint_objective(x_cdnt, region, v, demands, lambdas):
    xi, vi = categorize_x(x_cdnt, demands, lambdas, v)
    return v[0]*linalg.norm(x_cdnt - xi.get_cdnt()) + vi


def constraint_func(lambdas, demands, v, region: Region) -> optimize.NonlinearConstraint:
    objective = lambda x_cdnt: constraint_objective(x_cdnt, region, v, demands, lambdas)
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region.radius)
    result = optimize.minimize(objective, x0=np.zeros(2), method='SLSQP', constraints=x_in_R_constraint)
    return result.fun


def minimize_problem7(lambdas: list[float], demands: list[Demand], t: float, region: Region) -> list[float]:
    constraints = [optimize.NonlinearConstraint(lambda v: constraint_func(lambdas, demands, v, region), 0, np.inf)]
    objective = lambda v, demands, lambdas, t, region: objective_function(demands, lambdas, t, v, region)
    result = optimize.minimize(objective, x0=np.ones(demands.size + 1), args=(demands, lambdas, t, region), method='SLSQP', constraints=constraints)
    return result.x, result.fun


# x, fun_val = minimize_problem7(lambdas_temporary, demands, t_temporary, region)