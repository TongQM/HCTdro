import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import integrate, linalg

from findWorstTSPDensity import findWorstTSPDensity


def show_density(f_tilde, resolution=100, t=1, dmd_index=0):
    f = f_tilde
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')

    # Create the mesh in polar coordinates and compute corresponding Z.
    r = np.linspace(0, region.radius, resolution)
    p = np.linspace(0, 2*np.pi, resolution)
    R, P = np.meshgrid(r, p)
    f_vectorized = np.vectorize(f)
    Z = f_vectorized(R, P)

    # Plot the surface.
    plt.grid(False)
    # ax.pcolormesh(P, R, Z, cmap=plt.colormaps['Greys'])
    # actual plotting
    ctf = ax.contourf(P, R, Z, cmap=cm.binary)
    ax.scatter(generator.thetas, generator.rs, c='yellow')

    # add color bar
    plt.colorbar(ctf)

    # Tweak the limits and add latex math labels.
    # ax.set_xlabel(r'$X$')
    # ax.set_ylabel(r'$Y$')

    fig.show()
    # plt.savefig(f'figs/{dmd_index}-{t}.png')


# np.random.seed(11)
region = Region(1)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 5)
t, epsilon = 0.3, 1
tol = 1e-4

plot_resolution = 100


demands = generator.generate()
demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
f_tilde = findWorstTSPDensity(region, demands, t, epsilon, tol)
f_tilde_area, error = integrate.dblquad(lambda r, theta: r*f_tilde(r, theta), 0, 2*np.pi, lambda _: 0, lambda _: region.radius)
print(f'The measure of the whole region is {f_tilde_area}.')
show_density(f_tilde, plot_resolution, t)

