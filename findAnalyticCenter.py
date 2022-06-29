import numpy as np
import pandas as pd
from scipy import optimize


def find_analytic_center(objective, constraints, lambda_bounds, lambda0):
    result = optimize.minimize(objective, lambda0, method='SLSQP', bounds=lambda_bounds, constraints=constraints)
    return result.x, result.fun
