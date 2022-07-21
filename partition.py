import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from classes import Coordinate, Region, Demands_generator, Polyhedron
from scipy import integrate, linalg
from findWorstTSPDensity import findWorstTSPDensity
from python_tsp.distances  import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming


region = Region(1)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 5)
t, epsilon = 0.3, 1
tol = 1e-4


grids = np.linspace(0, 2*np.pi, 5, endpoint=True)
generator = Demands_generator(region, 30)
demands = generator.generate()


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


for i in grids:
    for j in grids:
        partition = [0, min(i, j), max(i, j), 2*np.pi]
        M = len(partition) - 1
        demands_partition = partition_demands(demands, partition)
        districts = [District((partition[k], partition[k+1]), demands_partition[k]) for k in range(M)]
        districts_tsp = [dist.find_optimal_tsp_solution() for dist in districts]



# def calculate_fixed_route_cost(cost, beta, demands, f_tilde):
