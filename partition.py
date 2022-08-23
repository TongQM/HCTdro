import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import integrate, linalg
from findWorstTSPDensity import findWorstTSPDensity
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search


region = Region(1)
depot = Coordinate(2, 0.3)
t, epsilon = 0.3, 1
tol = 1e-4


grids = np.linspace(0, 2*np.pi, 10, endpoint=False)
generator = Demands_generator(region, 20)
demands = generator.generate()


class Grids:
    def __init__(self, num_grid):
        self.num_grid = num_grid
        self.grids = np.linspace(0, 2*np.pi, self.num_grid, endpoint=False)     


class Partition:
    def __init__(self, num_districts, location):
        self.num_districts = num_districts
        self.location = location

    def next_partition(self):
        return


class District:
    def __init__(self, boundary, demands_within):
        self.boundary = boundary
        self.demands_within = demands_within
        self.demands_within_locations = [dmd.get_cdnt() for dmd in self.demands_within]

    def find_optimal_tsp_solution(self):
        self.distance_matrix = euclidean_distance_matrix(self.demands_within_locations)
        self.permutation, self.distance = solve_tsp_dynamic_programming(self.distance_matrix)
        return self.permutation, self.distance

    def __repr__(self) -> str:
        return f'{self.boundary}'

    def __str__(self) -> str:
        return self.__repr__()


def district_indicator(demand, partition):
    for ind in range(len(partition)-1):
        if demand.location.theta >= partition[ind] and demand.location.theta <  partition[ind+1]:
            return ind, partition[ind], partition[ind+1]


def partition_demands(demands, partition):
    M = len(partition) - 1
    demands_partition = [[] for _ in range(M)]
    for demand in demands: 
        demands_partition[district_indicator(demand, partition)[0]].append(demand)
    return demands_partition


for i in range(1, len(grids)):
    for j in range(i+1, len(grids)):
        partition = [0, grids[i], grids[j], 2*np.pi]
        M = len(partition) - 1
        demands_partition = partition_demands(demands, partition)
        districts = [District((partition[k], partition[k+1]), demands_partition[k]) for k in range(M)]
        districts_tsp = [dist.find_optimal_tsp_solution() for dist in districts]


# def calculate_fixed_route_cost(cost, beta, demands, f_tilde):
