from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator, Demand


def modified_norm(x_cdnt: list[float], i: int, demands: list[Demand], lambdas: list[float]) -> float:
    return linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas[i]


def integrand(r: float, theta: float, lambdas: list[float], v: list[float], demands: list[Demand]) -> float:
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    return np.sum([region_indicator(i, x_cdnt, lambdas, demands) / (v[0]*linalg.norm(x_cdnt - demands[i].get_cdnt()) + v[i+1]) for i in range(len(demands))])


def objective_function(demands: list[Demand], lambdas: list[float], t: float, v: list[float], region: Region) -> float:
    sum_integral, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(lambdas, v, demands), epsabs=1e-3)
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])


def region_indicator(i: int, x_cdnt: list[float], lambdas: list[float], demands: list[Demand]) -> Literal[0, 1]:
    for j in range(len(lambdas)):
        if modified_norm(x_cdnt, i, demands, lambdas) > modified_norm(x_cdnt, j, demands, lambdas):
            return 0
    return 1

def categorize_x(x_cdnt: list[float], demands: list[Demand], lambdas: list[float], v: list[float]):
    for i in len(demands):
        if region_indicator(i, x_cdnt, lambdas, demands): return v[i+1]


def constraint_func(lambdas, demands, v) -> optimize.NonlinearConstraint:
    objective = lambda x_cdnt: v[0]*linalg.norm(x_cdnt - demands[i].get_cdnt()) + np.sum([region_indicator(i, x_cdnt, lambdas, demands)*v])
    optimize.minimize()


def minimize_problem7(lambdas: list[float], demands: list[Demand], v: list[float], t: float, region: Region) -> list[float]:
    constraints = []
    objective = lambda v, demands, lambdas, t, region: objective_function(demands, lambdas, t, v, region)
    optimize.minimize(objective, x0=np.ones(demands.size + 1), args=(demands, lambdas, t, region), method='SLSQP', constraints=constraints)
    return