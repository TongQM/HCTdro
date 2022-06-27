import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from problem14 import *
from optimize import Coordinate, Region, Demands_generator
from scipy import optimize

def findWorstTSPDensity(Rg: Region, demands, t: float=10e-2, epsilon: float=10e-5):
    '''
    Algorithm by Carlsson, Behroozl, and Mihic, 2018.
    Code by Yidi Miao, 2022.

    This algorithm (find the worst TSP density) takes as input a compact planar region containing 
    a set of n distinct points, a distance threshold t, and a tolerance epsilon.

    Input: A compact, planar region Rg containing a set of distinct points x1, x2,..., xn, which are 
    interpreted as an empirical distribution f_hat, a distance parameter t, and a tolerance epsilon.

    Output: An epsilon-approximation of the distribution f* that maximizes iint_Rg sqrt(f(x)) dA 
    subject to the constraint that D(f_hat, f) <= t.

    This is a standard analytic center cutting plane method applied to problem (13), which has an 
    n-dimensional variable space.
    '''

    n = demands.size
    UB, LB = np.inf, -np.inf
    Lambda = [optimize.LinearConstraint(np.ones(n).T, 0, 0)]
    lambda_bounds = optimize.Bounds(-np.inf, Rg.diam)
    lambda0 = np.array([0, 0])
    # while (UB - LB > epsilon):
    lambda_bar, lambda_bar_func_val = find_analytic_center(lambda x: -np.sum(np.log(Rg.diam - x)), Lambda, lambda_bounds, lambda0)

    '''Build an upper bounding f_bar for the original problem (4)'''


    return lambda_bar, lambda_bar_func_val


def find_analytic_center(objective, constraints, lambda_bounds, lambda0):
    result = optimize.minimize(objective, lambda0, method='SLSQP', bounds=lambda_bounds, constraints=constraints)
    return result.x, result.fun

region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 2)
demands = generator.generate()
lambda_bar, lambda_bar_func_value = findWorstTSPDensity(region, demands)