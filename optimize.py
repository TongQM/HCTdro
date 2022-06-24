import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, cos, sin


class Coordinate:
    def __init__(self, r: float, rad: float):
        self.r = r
        self.rad = rad
        self.x_cd = self.r * cos(self.rad)
        self.y_cd = self.r * sin(self.rad)

    def __repr__(self):
        return f'Polar: ({self.r}, {self.rad}) ' + f'|X-Y Plane: ({self.x_cd}, {self.y_cd})'

    def __str__(self):
        return self.__repr__()

class Region:
    def __init__(self, radius: float):
        self.radius = radius
        self.diam = 2*radius

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return self.__repr__()

class Partition:
    def __init__(self, region: Region, depot: Coordinate, boundaries):
        self.region = region
        self.depot = depot
        self.boundaries = boundaries

class Demand:
    def __init__(self, location: Coordinate, dmd: float):
        self.location = location
        self.dmd = dmd

    def __repr__(self):
        return self.location.__repr__()

    def __str__(self):
        return self.location.__str__()


class Demands_generator:
    def __init__(self, region: Region, Num_demands_pts: int):
        self.region = region
        self.Num_demands_pts = Num_demands_pts

    def generate(self):
        rs = np.random.uniform(low=0, high=self.region.radius, size=self.Num_demands_pts)
        rads = np.random.uniform(low=0, high=2*pi, size=self.Num_demands_pts)
        demands = np.array([Demand(Coordinate(rs[k], rads[k]), 1) for k in range(self.Num_demands_pts)])
        return demands
        
class Solution:
    def __init__(self, region: Region, demands, routes):
        self.region = region
        self.demands = demands
        self.routes = routes

    def evaluate(self):
        return 0


region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 100)
demands = generator.generate()
rds, rads = [d.location.r for d in demands], [d.location.rad for d in demands]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(rds, rads)