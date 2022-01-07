import bempp.api
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



k = 1
c = 1500
rho = 1000
omega = c*k

grid = bempp.api.shapes.sphere(h=0.1)
piecewise_const_space = bempp.api.function_space(grid, "DP", 0)

@bempp.api.complex_callable
def f(x, n, domain_index, result):
    result[0] = 1

vn = bempp.api.GridFunction(piecewise_const_space, fun=f)
