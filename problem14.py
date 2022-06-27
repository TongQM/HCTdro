from urllib.parse import _ResultMixinStr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate, linalg
from optimize import Region, Coordinate, Demands_generator

region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 10)
demands = generator.generate()
lambdas_temporary = np.ones(10)
t_temporary = 12
v0_temporary = 1
v1_temporary = 1
rds, rads = [d.location.r for d in demands], [d.location.rad for d in demands]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(rds, rads)


def integrand(r: float, theta: float, v0, v1, demands, lambdas, t):
    # Calculate a list of ||x-xi|| - lambda_i
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    modified_norms = [linalg.norm(x_cdnt - demands[i].get_cdnt()) - lambdas[i] for i in range(len(demands))]
    intgrd = 1/(4*(v0*np.min(modified_norms) + v1))
    return (intgrd + v0*t + v1)*r    # Jacobian

def objective_function(demands, lambdas, t, v0, v1, region: Region):
    penalty = v0
    area, error = integrate.dblquad(integrand, 0, 2*np.pi, lambda theta: 0, lambda theta: region.radius, args=(v0, v1, demands, lambdas, t))
    return area, error

def penalty(demands, lambdas, t, v0, v1, region: Region):
    hf_penalty_x, hf_penalty_fun = hf_penalty_objective(demands, lambdas)
    penalty_val = v0*hf_penalty_fun + v1
    return penalty_val

def hf_penalty_objective(demands, lambdas):
    n = demands.size
    result = optimize.minimize(lambda x: np.min([linalg.norm(x - demands[i].get_cdnt()) - lambdas[i] for i in range(n)]), x0 = np.zeros(2), method='Nelder-Mead')
    return result.x, result.fun

def minimize_problem14(demands, lambdas, t, region: Region):
    nu = 10 # penalty factor
    objective = lambda v, demands, lambdas, t, region: objective_function(demands, lambdas, t, v[0], v[1], region)[0] - nu*penalty(demands, lambdas,t, v[0], v[1], region)
    result = optimize.minimize(objective, x0=np.array([0, 0]), args=(demands, lambdas, t, region), method='Nelder-Mead')
    return result.x, result.fun


v, error = minimize_problem14(demands, lambdas_temporary, t_temporary, region)
# area, error = objective_function(demands, lambdas_temporary, t_temporary, v0_temporary, v1_temporary, region)