import numpy as np
from scipy import optimize, integrate

def findWassersteinDistance(demands, f_hat):
    lambda_star, 