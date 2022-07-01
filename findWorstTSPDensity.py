import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from problem14 import minimize_problem14
from problem7 import minimize_problem7, constraint_func, categorize_x, region_indicator
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import optimize, integrate, linalg

def findWorstTSPDensity(region: Region, demands, t: float=10e-2, epsilon: float=0.1):
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
    lambdas_bar = np.zeros(n)
    polyhedron = Polyhedron(np.eye(n), region.diam*np.ones(n), np.ones((1, n)), 0, n)
    while (UB - LB > epsilon):
        lambdas_bar, lambdas_bar_func_val = polyhedron.find_analytic_center(lambdas_bar)

        '''Build an upper bounding f_bar for the original problem (4).'''
        v_bar, problem14_func_val = minimize_problem14(demands, lambdas_bar, t, region)
        upper_integrand = lambda r, theta, demands, lambdas_bar, v_bar: r*np.sqrt(f_bar(r, theta, demands, lambdas_bar, v_bar))
        UB = integrate.dblquad(upper_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands, lambdas_bar, v_bar), epsabs=1e-3)

        '''Build an lower bounding f_tilde that us feasible for (4) by construction.'''
        v_tilde, problem7_func_val = minimize_problem7(lambdas_bar, demands, t, region)
        lower_integrand = lambda r, theta, demands, lambdas_bar, v_tilde: r*np.sqrt(f_tilde(r, theta, demands, lambdas_bar, v_tilde))
        LB = integrate.dblquad(lower_integrand, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands, lambdas_bar, v_tilde), epsabs=1e-3)

        '''Update g.'''
        g = np.zeros(len(demands))
        for i in range(len(demands)):
            integrandi = lambda r, theta, demands, lambdas_bar, v_bar: r*region_indicator(i, np.array([r*np.cos(theta), r*np.sin(theta)]), lambdas_bar, demands)*f_bar(r, theta, demands, lambdas_bar, v_bar) 
            g[i] = integrate.dblquad(integrandi, 0, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(demands, lambdas_bar, v_bar))

        '''Update polyheron Lambda to get next analytic center.'''
        polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)

    return lambda r, theta: f_tilde(r, theta, demands, lambdas_bar, v_tilde)


def f_bar(r, theta, demands, lambdas_bar, v_bar):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    return pow(1/4 * (v_bar[0]*np.min([linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas_bar[i] for i in range(len(demands))]) + v_bar[1]), -2)

def f_tilde(r, theta, demands, lambdas_bar, v_tilde):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    return 1/4 * np.sum([region_indicator(i, x_cdnt, lambdas_bar, demands) * pow((v_tilde[0]*linalg.norm(x_cdnt - demands[i].get_cdnt()) + v_tilde[i+1]), -2) for i in range(len(demands))])


region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 2)
demands = generator.generate()
f = findWorstTSPDensity(region, demands)