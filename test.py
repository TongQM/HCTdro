from kiwisolver import Constraint
from scipy import optimize
import numpy as np

def func(x: np.array):
    return -x.T @ x


x0 = np.array([1, -1])
constraints = []
constraint1 = optimize.LinearConstraint(np.ones(2), 0, 0)
constraint2 = optimize.NonlinearConstraint(lambda x: np.array([1,2,3,4,5,6,7,8,9,10]).T @ x, -2, 4)
constraints.extend([constraint1])
result = optimize.minimize(lambda x: -x[0]**2 -x[1]**2, x0, method='SLSQP', constraints=constraints)
