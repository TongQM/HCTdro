import numpy as np
import time
import numba as nb
from scipy import integrate
from problem7 import categorize_x
from classes import Region, Coordinate, Demands_generator


class Integral:
    
    def __init__(self, function, lower_bound, upper_bound, lower_func, upper_func):
        self.function = function
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.lower_func, self.upper_func = lower_func, upper_func

    def MCI(self, N):
        start_time = time.time()
        theta_samples = np.random.uniform(self.lower_bound, self.upper_bound, N)
        r_samples = np.random.uniform(self.lower_func(1), self.upper_func(1), N)
        samples_vals = np.array(list(map(self.function, r_samples, theta_samples)))
        area =  100*np.pi * np.mean(samples_vals)
        end_time = time.time()
        return area, end_time - start_time

    def sci_integral(self):
        start_time = time.time()
        area, error = integrate.dblquad(lambda r, theta: r*self.function(r, theta), self.lower_bound, self.upper_bound, self.lower_func, self.upper_func)
        end_time = time.time()
        return area, end_time - start_time

    def compare(self, N):
        scipy_integral = self.sci_integral()
        MCI = self.MCI(N)
        print(f'Scipy-integrate: {scipy_integral[0]} with time {scipy_integral[1]}s; Monte Carlo: {MCI[0]} with {MCI[1]}s.')

@nb.njit
def integrand(r: float, theta: float, lambdas: list[float], v: list[float], demands_locations: list[list[float]]) -> float:
    x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
    xi, vi = categorize_x(x_cdnt, demands_locations, lambdas, v)
    raw_intgrd = 1 / (v[0]*np.linalg.norm(x_cdnt - xi) + vi)
    return np.sqrt(raw_intgrd)

region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 5)
demands = generator.generate()
n = demands.shape[0]
lambdas_bar = np.zeros(n)
demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
v = np.array([2.80058103, 1.25083629, 1.87593587, 1.86719344, 1.52163244, 2.05530126])
objective = lambda r, theta: integrand(r, theta, lambdas_bar, v, demands_locations)
integral = Integral(objective, 0, 2*np.pi, lambda _: 0, lambda _: region.radius)
