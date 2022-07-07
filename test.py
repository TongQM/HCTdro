from scipy import integrate
import numpy as np
def f(y, x):
    return x * y
def h(x):
    return x
v, err = integrate.dblquad(f, 1, 2, lambda x: 1, h)
print(v)