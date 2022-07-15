import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import numba as nb
from problem14 import minimize_problem14, min_modified_norm
from problem7 import minimize_problem7, constraint_func, categorize_x, region_indicator, norm_func
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import optimize, integrate, linalg


def findWorstTSPDensity(region: Region, demands, t: float=1, epsilon: float=0.1, tol: float=1e-4):
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

    np.random.seed(11)
    n = demands.shape[0]
    UB, LB = np.inf, -np.inf
    lambdas_bar = np.zeros(n)
    polyhedron = Polyhedron(np.eye(n), region.diam*np.ones(n), np.ones((1, n)), 0, n)
    k = 1
    while (abs(UB - LB) > epsilon):
        print(f'Iteration {k} begins: \n')
        starttime = time.time()
        lambdas_bar, lambdas_bar_func_val = polyhedron.find_analytic_center(lambdas_bar)
        time1 = time.time()
        print(f'Find analytic center: Lambdas_bar is {lambdas_bar}, with value {lambdas_bar_func_val}, took {time1 - starttime}s.')

        demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])

        '''Build an upper bounding f_bar for the original problem (4).'''
        v_bar, problem14_func_val = minimize_problem14(demands, lambdas_bar, t, region.radius)
        upper_integrand = lambda r, theta: r*np.sqrt(f_bar(r, theta, demands_locations, lambdas_bar, v_bar))
        UB, UB_error = integrate.dblquad(upper_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius,epsabs=tol)
        time2 = time.time()
        print(f'Find upper bound: Upper bound is {UB}, with error {UB_error}, took {time2 - time1}s.')

        '''Build an lower bounding f_tilde that us feasible for (4) by construction.'''
        v_tilde, problem7_func_val = minimize_problem7(lambdas_bar, demands, t, region.radius)
        lower_integrand = lambda r, theta, demands_locations, lambdas_bar, v_tilde: r*np.sqrt(f_tilde(r, theta, demands_locations, lambdas_bar, v_tilde))
        LB, LB_error = integrate.dblquad(lower_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands_locations, lambdas_bar, v_tilde), epsabs=tol)
        time3 = time.time()
        print(f'Find lower bound: Lower bound is {LB}, with error {LB_error}, took {time3 - time2}s.')

        '''Update g.'''
        g = np.zeros(len(demands))
        for i in range(len(demands)):
            integrandi = lambda r, theta, demands, lambdas_bar, v_bar: r*region_indicator(i, np.array([r*np.cos(theta), r*np.sin(theta)]), lambdas_bar, demands_locations)*f_bar(r, theta, demands, lambdas_bar, v_bar) 
            g[i], g_error = integrate.dblquad(integrandi, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands_locations, lambdas_bar, v_bar), epsabs=tol)

        '''Update polyheron Lambda to get next analytic center.'''
        polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)
        time4 = time.time()
        print(f'It took {time4 - time3}s to get vector g.')

        endtime = time.time()
        print(f'End of iteration {k}.\n  The whole iteration took {endtime - starttime}s.\n')
        k += 1

    return lambda r, theta: f_tilde(r, theta, demands_locations, lambdas_bar, v_tilde)


@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@nb.njit
def f_bar(r, theta, demands_locations, lambdas_bar, v_bar):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    return 1/4 * pow((v_bar[0]*min_modified_norm(x_cdnt, demands_locations, lambdas_bar) + v_bar[1]), -2)

@nb.njit
def f_tilde(r, theta, demands_locations, lambdas_bar, v_tilde):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    xi, vi = categorize_x(x_cdnt, demands_locations, lambdas_bar, v_tilde)
    return 1/4 * pow(v_tilde[0] * norm_func(x_cdnt, xi) + vi, -2)


'''
Algorithm optimization:
Store the division of xs: HOW?
Add Jacobian to minimize: save time of numerical estimation of gradient
Improve integration method: CORE problem
'''

# np.random.seed(11)
# region = Region(10)
# depot = Coordinate(2, 0.3)
# generator = Demands_generator(region, 5)
# demands = generator.generate()
# t, epsilon = 3.5, 1
# n = demands.shape[0]
# UB, LB = np.inf, -np.inf
# tol = 1e-5
# lambdas_bar = np.zeros(n)
# polyhedron = Polyhedron(np.eye(n), region.diam*np.ones(n), np.ones((1, n)), 0, n)
# k = 1
# while (abs(UB - LB) > epsilon):
#     print(f'Iteration {k} begins: \n')
#     starttime = time.time()
#     lambdas_bar, lambdas_bar_func_val = polyhedron.find_analytic_center(lambdas_bar)
#     time1 = time.time()
#     print(f'Find analytic center: Lambdas_bar is {lambdas_bar}, with value {lambdas_bar_func_val}, took {time1 - starttime}s.')

#     demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])

#     '''Build an upper bounding f_bar for the original problem (4).'''
#     v_bar, problem14_func_val = minimize_problem14(demands, lambdas_bar, t, region.radius)
#     upper_integrand = lambda r, theta: r*np.sqrt(f_bar(r, theta, demands_locations, lambdas_bar, v_bar))
#     UB, UB_error = integrate.dblquad(upper_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius,epsabs=tol)
#     time2 = time.time()
#     print(f'Find upper bound: Upper bound is {UB}, with error {UB_error}, took {time2 - time1}s.')

#     '''Build an lower bounding f_tilde that us feasible for (4) by construction.'''
#     v_tilde, problem7_func_val = minimize_problem7(lambdas_bar, demands, t, region.radius)
#     lower_integrand = lambda r, theta, demands_locations, lambdas_bar, v_tilde: r*np.sqrt(f_tilde(r, theta, demands_locations, lambdas_bar, v_tilde))
#     LB, LB_error = integrate.dblquad(lower_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands_locations, lambdas_bar, v_tilde), epsabs=tol)
#     time3 = time.time()
#     print(f'Find lower bound: Lower bound is {LB}, with error {LB_error}, took {time3 - time2}s.')

#     '''Update g.'''
#     g = np.zeros(len(demands))
#     for i in range(len(demands)):
#         integrandi = lambda r, theta, demands, lambdas_bar, v_bar: r*region_indicator(i, np.array([r*np.cos(theta), r*np.sin(theta)]), lambdas_bar, demands_locations)*f_bar(r, theta, demands, lambdas_bar, v_bar) 
#         g[i], g_error = integrate.dblquad(integrandi, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands_locations, lambdas_bar, v_bar), epsabs=tol)

#     '''Update polyheron Lambda to get next analytic center.'''
#     polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)
#     time4 = time.time()
#     print(f'It took {time4 - time3}s to get vector g.')

#     endtime = time.time()
#     print(f'End of iteration {k}.\n  The whole iteration took {endtime - starttime}s.\n')
#     k += 1




'''
Plots:
5 points plot, density + empirical points (hyperparameters: distribution of 5 points, Wasserstein distance bound 1, 2, 3, 4, 5, 5*5 runs)
'''