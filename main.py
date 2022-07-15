import numpy as np
import matplotlib.pyplot as plt
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import integrate, linalg

from findWorstTSPDensity import findWorstTSPDensity



np.random.seed(11)
region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 5)
demands = generator.generate()
t, epsilon = 3.5, 1
tol = 1e-4
demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])

f_tilde = findWorstTSPDensity(region, demands, t, epsilon, tol)
f_tilde_area, error = integrate.dblquad(lambda r, theta: r*f_tilde(r, theta), 0, 2*np.pi, lambda _: 0, lambda _: region.radius)
print(f'The measure of the whole region is {f_tilde_area}.')


# def show_density(f_tilde):
f = f_tilde
fig = plt.figure()
ax = fig.add_subplot(projection='polar')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, region.radius, 500)
p = np.linspace(0, 2*np.pi, 500)
R, P = np.meshgrid(r, p)
f_vectorized = np.vectorize(f)
Z = f_vectorized(R, P)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
plt.grid(False)
ax.pcolormesh(P, R, Z)
ax.scatter(generator.thetas, generator.rs)

# Tweak the limits and add latex math labels.
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')

plt.show()