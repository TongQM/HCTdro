import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate, linalg
from optimize import Region, Coordinate, Demands_generator

n = 2
np.random.seed(11)
region = Region(2)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, n)
demands = generator.generate()
lambdas_temporary = np.zeros(n)
t_temporary = 1
v0_temporary = 1
v1_temporary = 1
rs, thetas = [d.location.r for d in demands], [d.location.theta for d in demands]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(thetas, rs)


def modified_min_norm(x_cdnt, demands, lambdas):
    return np.min([linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas[i] for i in range(len(demands))])

def integrand(r: float, theta: float, v, demands, lambdas):
    # Calculate a list of ||x-xi|| - lambda_i
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    raw_intgrd = 1/(4*(v[0]*modified_min_norm(x_cdnt, demands, lambdas) + v[1]))
    return raw_intgrd*r    # r as Jacobian

def objective_function(demands, lambdas, t, v, region: Region):
    area, error = integrate.dblquad(integrand, 0.001, 2*np.pi, lambda _: 0, lambda _: region.radius, args=(v, demands, lambdas), epsabs=1e-3)
    return area + v[0]*t + v[1], error

def constraint_coeff(demands, lambdas, region: Region):
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region.radius)
    result = optimize.minimize(lambda x_cdnt: modified_min_norm(x_cdnt, demands, lambdas), x0 = np.ones(2), method='SLSQP', constraints=x_in_R_constraint)
    return np.array([result.fun, 1])

def minimize_problem14(demands, lambdas, t, region: Region):
    constraints = [optimize.LinearConstraint(constraint_coeff(demands, lambdas, region), 0, np.inf)]
    bound = optimize.Bounds(0, np.inf)
    objective = lambda v, demands, lambdas, t, region: objective_function(demands, lambdas, t, v, region)[0]
    result = optimize.minimize(objective, x0=np.ones(2), args=(demands, lambdas, t, region), method='SLSQP', bounds=bound, constraints=constraints)
    return result.x, result.fun


v, func_value = minimize_problem14(demands, lambdas_temporary, t_temporary, region)