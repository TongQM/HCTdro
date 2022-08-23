import numpy as np
import numba as nb
from scipy import optimize, integrate
from problem14 import min_modified_norm, tol

@nb.njit
def integrand(r, theta, demands_locations, f, lambdas):
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    return r * f(r, theta) * min_modified_norm(x_cdnt, demands_locations, lambdas)

def objective_function(lambdas, demands_locations, f, region_radius):
    area, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda _: 0, lambda _: region_radius, args=(demands_locations, f, lambdas), epsabs=tol)
    return -area

def findWassersteinDistance(demands_locations, f, region_radius):
    n = len(demands_locations)
    constraints_dict = {'type': 'eq', 'fun': lambda lambdas: np.ones(n) @ lambdas, 'jac': lambda _: np.ones(n)}
    lambdas_star, lambdas_fun_val = optimize.minimize(objective_function, x0=np.zeros(len(demands_locations)), arg=(demands_locations, f, region_radius), method='SLSQP',constraints=constraints_dict)
    return lambdas_fun_val