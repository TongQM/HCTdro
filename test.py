from kiwisolver import Constraint
from scipy import optimize
import numpy as np

def func(x: np.array):
    return x.T @ x


x0 = np.ones(10)
constraints = []
constraint1 = optimize.LinearConstraint(np.ones(10), 0, 0)
constraint2 = optimize.NonlinearConstraint(lambda x: np.array([1,2,3,4,5,6,7,8,9,10]).T @ x, -2, 4)
constraints.extend([constraint1, constraint2])
result = optimize.minimize(func, x0, method='SLSQP', constraints=constraints)
