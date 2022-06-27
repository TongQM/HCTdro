import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate, linalg
from optimize import Region, Coordinate, Demands_generator

n = 10
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


def integrand(r: float, theta: float, v0, v1, demands, lambdas, t):
    # Calculate a list of ||x-xi|| - lambda_i
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    modified_norms = [linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas[i] for i in range(n)]
    # print(f'v0: {v0} | v1: {v1} | min: {np.min(modified_norms)}')
    intgrd = 1/(4*(v0*np.min(modified_norms) + v1))
    return intgrd*r    # r as Jacobian

def objective_function(demands, lambdas, t, v0, v1, region: Region):
    area, error = integrate.dblquad(integrand, 0.001, 2*np.pi, lambda theta: 0, lambda theta: region.radius, args=(v0, v1, demands, lambdas, t), epsabs=1e-3)
    return area + v0*t + v1, error

def hf_penalty_objective(demands, lambdas, region: Region):
    x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region.radius)
    result = optimize.minimize(lambda x: np.min([linalg.norm(x - demands[i].get_cdnt()) - lambdas[i] for i in range(n)]), x0 = np.ones(2), method='SLSQP', constraints=x_in_R_constraint)
    return result.fun

def minimize_problem14(demands, lambdas, t, region: Region):
    v0_coeff = hf_penalty_objective(demands, lambdas, region)
    # print(f'v0_coeff: {v0_coeff}')
    constraint1 = [optimize.LinearConstraint(np.array([v0_coeff, 1]), 0, np.inf)]
    bound1 = optimize.Bounds(0, np.inf)
    objective = lambda v, demands, lambdas, t, region: objective_function(demands, lambdas, t, v[0], v[1], region)[0]
    result = optimize.minimize(objective, x0=np.ones(2), args=(demands, lambdas, t, region), method='SLSQP', bounds=bound1, constraints=constraint1)
    return result.x, result.fun, v0_coeff


v, func_value, v0_coeff = minimize_problem14(demands, lambdas_temporary, t_temporary, region)
# area, error = objective_function(demands, lambdas_temporary, t_temporary, v0_temporary, v1_temporary, region)