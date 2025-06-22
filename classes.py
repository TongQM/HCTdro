from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import gurobipy as gp
from math import pi, cos, sin
import os


def append_df_to_csv(filename, df, sep=",", header=True, index=False):
    """
    Append a DataFrame [df] to a CSV file [filename].
    If [filename] doesn't exist, this function will create it.

    This function also prints the number of rows in the existing CSV file
    before appending the new data.

    Parameters:
      filename : String. File path or existing CSV file
                 (Example: '/path/to/file.csv')
      df : DataFrame to save to CSV file
      sep : String. Delimiter to use, default is comma (',')
      header : Boolean or list of string. Write out the column names. If a list of strings
               is given it is assumed to be aliases for the column names
      index : Boolean. Write row names (index)
    """
    # Check if file exists
    file_exists = os.path.isfile(filename)

    if file_exists:
        # Read the existing CSV to find the number of rows
        existing_df = pd.read_csv(filename, sep=sep)
        # print(f"Number of rows in existing CSV: {len(existing_df)}")

        # Append without header
        df.to_csv(filename, mode='a', sep=sep, header=False, index=index)
    else:
        # If file doesn't exist, create it with header
        df.to_csv(filename, mode='w', sep=sep, header=header, index=index)
        # print("Created new CSV file.")


class Coordinate:
    def __init__(self, r: float, theta: float):
        self.r = r
        self.theta = theta
        self.x_cd = self.r * cos(self.theta)
        self.y_cd = self.r * sin(self.theta)

    def __repr__(self):
        return f'Polar: (r: {self.r}, theta: {self.theta}) ' + f'| X-Y Plane: ({self.x_cd}, {self.y_cd})'

    def __str__(self):
        return self.__repr__()

class Region:
    def __init__(self, radius: float):
        self.radius = radius
        self.diam = 2*radius

    def __repr__(self) -> str:
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.scatter([], [])
        # plt.show()
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
    def __init__(self, region: Region, Num_demands_pts: int, seed=11):
        self.region = region
        self.Num_demands_pts = Num_demands_pts
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        self.rs = np.random.uniform(low=0, high=self.region.radius, size=self.Num_demands_pts)
        self.thetas = np.random.uniform(low=0, high=2*pi, size=self.Num_demands_pts)
        demands = np.array([Demand(Coordinate(self.rs[k], self.thetas[k]), 1) for k in range(self.Num_demands_pts)])
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
        self.eq_constraints = {'type': 'eq', 'fun': lambda x: self.B @ x - self.c , 'jac': lambda _: self.B}
        # optimize.LinearConstraint(B, c, c)
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x + 1e-6, 'jac': lambda _: -self.A}
        # optimize.LinearConstraint(A, -np.inf, b + 1e-4, keep_feasible=False)

    def add_ineq_constraint(self, ai, bi):
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.ineq_constraints = optimize.LinearConstraint(self.A, -np.inf, self.b)

    def find_analytic_center(self, x0):
        # Find a feasible solution to the problem first
        find_feasible_sol = gp.Model('find_feasible_sol')
        find_feasible_sol.setParam('OutputFlag', 1)
        x = find_feasible_sol.addMVar(shape=self.dim, lb=-1, ub=1, name='x')
        find_feasible_sol.addConstr(self.B @ x == self.c)
        find_feasible_sol.addConstr(self.A @ x <= self.b)
        find_feasible_sol.setObjective(0, gp.GRB.MINIMIZE)
        find_feasible_sol.optimize()
        # assert find_feasible_sol.status == gp.GRB.OPTIMAL, find_feasible_sol.status
        x0 = x.X

        objective = lambda x: -np.sum(np.log(self.b - self.A @ x + 1e-6))  # To ensure log(b - A @ x) is defined.
        objective_jac = lambda x: np.sum(np.array([self.A[i, :]/(self.b[i] - self.A[i, :] @ x) for i in range(self.A.shape[0])]), axis=0)
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=[self.ineq_constraints, self.eq_constraints], jac='cs', options={'maxiter': 1000,'disp': True})
        assert result.success, result.message
        analytic_center, analytic_center_val = result.x, result.fun            
        return analytic_center, analytic_center_val

    def show_constraints(self):
        print(f'A: {self.A} \n b: {self.b} \n B: {self.B} \n c: {self.c}.')