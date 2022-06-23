import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose as mls
from math import pi


class Coordinate:
    def __init__(self, r: float, rad: float):
        self.r = r
        self.rad = rad

    def __repr__(self):
        return f'({self.r}, {self.rad})'

    def __str__(self):
        return self.__repr__()

class Region:
    def __init__(self, radius: float):
        self.radius = radius

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return self.__repr__()

class Partition:
    def __init__(self, region: Region, depot: Coordinate):
        self.region = region
        self.depot = depot

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
        demands = [Demand(Coordinate(rs[k], rads[k]), 1) for k in range(self.Num_demands_pts)]
        return demands
        

region = Region(10)
depot = Coordinate(2, 0.3)
generator = Demands_generator(region, 10)
demands = generator.generate()
