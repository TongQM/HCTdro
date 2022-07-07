import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from math import pi, cos, sin


class Coordinate:
    def __init__(self, r: float, theta: float):
        self.r = r
        self.theta = theta
        self.x_cd = self.r * cos(self.theta)
        self.y_cd = self.r * sin(self.theta)

    def __repr__(self):
        return f'Polar: (r: {self.r}, rad: {self.theta}) ' + f'| X-Y Plane: ({self.x_cd}, {self.y_cd})'

    def __str__(self):
        return self.__repr__()

class Region:
    def __init__(self, radius: float):
        self.radius = radius
        self.diam = 2*radius

    def __repr__(self) -> str:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter([], [])
        plt.show()
        return f'radius: {self.radius}'

    def __str__(self) -> str:
        return self.__repr__()

class Partition:
    def __init__(self, region: Region, depot: Coordinate, boundaries):
        self.region = region
        self.depot = depot
        self.boundaries = boundaries

class Demand:
    def __init__(self, location: Coordinate, dmd: float):
        self.location = location
        self.dmd = dmd

    def get_cdnt(self):
        return np.array([self.location.x_cd, self.location.y_cd])

    def __repr__(self):
        return self.location.__repr__()

    def __str__(self):
        return self.location.__str__()


class Demands_generator:
    def __init__(self, region: Region, Num_demands_pts: int):
        self.region = region
        self.Num_demands_pts = Num_demands_pts

    def generate(self):
        rs = np.random.uniform(low=0, high=self.region.radius, size=self.Num_demands_pts)
        thetas = np.random.uniform(low=0, high=2*pi, size=self.Num_demands_pts)
        demands = np.array([Demand(Coordinate(rs[k], thetas[k]), 1) for k in range(self.Num_demands_pts)])
        return demands
        
class Solution:
    def __init__(self, region: Region, demands, routes):
        self.region = region
        self.demands = demands
        self.routes = routes

    def evaluate(self):
        return 0


class Polyhedron:
    def __init__(self, A, b, B, c, dimension):
        '''
        Polyhedron determined by Ax<=b form and Bx=c
        '''
        self.A, self.b = A, b
        self.B, self.c = B, c
        self.dim = dimension
        self.eq_constraints = optimize.LinearConstraint(B, c, c)
        self.ineq_constraints = optimize.LinearConstraint(A, -np.inf, b + 1e-4, keep_feasible=False)

    def add_ineq_constraint(self, ai, bi):
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.ineq_constraints = optimize.LinearConstraint(self.A, -np.inf, self.b)

    def find_analytic_center(self, x0):
        objective = lambda x: -np.sum(np.log(self.b - self.A @ x + 1e-6))  # To ensure log(b - A @ x) is defined.
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=(self.ineq_constraints, self.eq_constraints))
        assert result.success, result.message
        analytic_center, analytic_center_val = result.x, result.fun            
        return analytic_center, analytic_center_val

    def show_constraints(self):
        print(f'A: {self.A} \n b: {self.b} \n B: {self.B}, c: {self.c}.')