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

# New: Cartesian version for square region
def min_modified_norm_cartesian(x_cdnt, demands_locations, lambdas):
    return np.min([np.linalg.norm(x_cdnt - demands_locations[i]) - lambdas[i] for i in range(len(demands_locations))])

def integrand_cartesian(x, y, demands_locations, f, lambdas):
    x_cdnt = np.array([x, y])
    return f(x, y) * min_modified_norm_cartesian(x_cdnt, demands_locations, lambdas)

def objective_function_cartesian(lambdas, demands_locations, f, region):
    area, error = integrate.dblquad(
        lambda y, x: integrand_cartesian(x, y, demands_locations, f, lambdas),
        region.x_min, region.x_max,
        region.y_min, region.y_max,
        epsabs=tol
    )
    return -area

def findWassersteinDistance_cartesian(demands_locations, f, region):
    n = len(demands_locations)
    constraints_dict = {'type': 'eq', 'fun': lambda lambdas: np.ones(n) @ lambdas, 'jac': lambda _: np.ones(n)}
    res = optimize.minimize(
        objective_function_cartesian,
        x0=np.zeros(n),
        args=(demands_locations, f, region),
        method='SLSQP',
        constraints=constraints_dict
    )
    return -res.fun