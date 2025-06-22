import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import gurobipy as gp
import os


def append_df_to_csv(filename, df, sep=",", header=True, index=False):
    """
    Append a DataFrame [df] to a CSV file [filename].
    If [filename] doesn't exist, this function will create it.

    This function also prints the number of rows in the existing CSV file
    before appending the new data.

    Parameters:
      filename : String. File path or existing CSV file
                 (Example: '/path/to/file.csv')
      df : DataFrame to save to CSV file
      sep : String. Delimiter to use, default is comma (',')
      header : Boolean or list of string. Write out the column names. If a list of strings
               is given it is assumed to be aliases for the column names
      index : Boolean. Write row names (index)
    """
    # Check if file exists
    file_exists = os.path.isfile(filename)

    if file_exists:
        # Read the existing CSV to find the number of rows
        existing_df = pd.read_csv(filename, sep=sep)
        # print(f"Number of rows in existing CSV: {len(existing_df)}")

        # Append without header
        df.to_csv(filename, mode='a', sep=sep, header=False, index=index)
    else:
        # If file doesn't exist, create it with header
        df.to_csv(filename, mode='w', sep=sep, header=header, index=index)
        # print("Created new CSV file.")


class CartesianGrid:
    """Cartesian grid for discretizing the square region"""
    
    def __init__(self, size: int, side_length: float = 1.0, center_x: float = 0.0, center_y: float = 0.0):
        self.size = size
        self.side_length = side_length
        self.center_x = center_x
        self.center_y = center_y
        self.x_min = center_x - side_length / 2
        self.x_max = center_x + side_length / 2
        self.y_min = center_y - side_length / 2
        self.y_max = center_y + side_length / 2
        
        # Create grid points
        self.x_coords = np.linspace(self.x_min, self.x_max, size + 1)
        self.y_coords = np.linspace(self.y_min, self.y_max, size + 1)
        
        # Cell centers
        self.cell_centers_x = (self.x_coords[:-1] + self.x_coords[1:]) / 2
        self.cell_centers_y = (self.y_coords[:-1] + self.y_coords[1:]) / 2
        
        # Cell area
        self.cell_area = (side_length / size) ** 2
        
        # Total number of cells
        self.num_cells = size ** 2
        
    def get_cell_center(self, i: int, j: int):
        """Get the center coordinates of cell (i, j)"""
        return self.cell_centers_x[i], self.cell_centers_y[j]
    
    def get_cell_index(self, i: int, j: int):
        """Convert 2D grid indices to 1D index"""
        return i * self.size + j
    
    def get_grid_indices(self, cell_index: int):
        """Convert 1D cell index to 2D grid indices"""
        i = cell_index // self.size
        j = cell_index % self.size
        return i, j


class EmpiricalDistribution:
    """Empirical distribution on the Cartesian grid"""
    
    def __init__(self, grid: CartesianGrid):
        self.grid = grid
        self.density = np.zeros(grid.num_cells)
        self.samples = []
        
    def generate_random_samples(self, num_samples: int, seed: int = None):
        """Generate random samples and create empirical distribution"""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random points in the region
        x_samples = np.random.uniform(self.grid.x_min, self.grid.x_max, num_samples)
        y_samples = np.random.uniform(self.grid.y_min, self.grid.y_max, num_samples)
        
        self.samples = list(zip(x_samples, y_samples))
        
        # Assign samples to grid cells
        for x, y in self.samples:
            # Find which cell this point belongs to
            i = int((x - self.grid.x_min) / (self.grid.side_length / self.grid.size))
            j = int((y - self.grid.y_min) / (self.grid.side_length / self.grid.size))
            
            # Ensure indices are within bounds
            i = min(max(i, 0), self.grid.size - 1)
            j = min(max(j, 0), self.grid.size - 1)
            
            cell_index = self.grid.get_cell_index(i, j)
            self.density[cell_index] += 1
            
    def normalize(self):
        """Normalize the density to sum to 1"""
        total = self.density.sum()
        if total > 0:
            self.density = self.density / total
            
    def set_density(self, density):
        """Set the density directly"""
        if len(density) == self.grid.num_cells:
            self.density = np.array(density)
        else:
            raise ValueError(f"Density must have length {self.grid.num_cells}")


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Cartesian: ({self.x}, {self.y})'

    def __str__(self):
        return self.__repr__()

    def get_coordinates(self):
        return np.array([self.x, self.y])


class SquareRegion:
    def __init__(self, side_length: float, center_x: float = 0.0, center_y: float = 0.0):
        self.side_length = side_length
        self.center_x = center_x
        self.center_y = center_y
        self.x_min = center_x - side_length / 2
        self.x_max = center_x + side_length / 2
        self.y_min = center_y - side_length / 2
        self.y_max = center_y + side_length / 2

    def __repr__(self) -> str:
        return f'Square: side_length={self.side_length}, center=({self.center_x}, {self.center_y})'

    def __str__(self):
        return self.__repr__()

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside the square region"""
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)


class Partition:
    def __init__(self, region: SquareRegion, depot: Coordinate, boundaries):
        self.region = region
        self.depot = depot
        self.boundaries = boundaries


class Demand:
    def __init__(self, location: Coordinate, dmd: float):
        self.location = location
        self.dmd = dmd

    def get_coordinates(self):
        return self.location.get_coordinates()

    def __repr__(self):
        return self.location.__repr__()

    def __str__(self):
        return self.location.__str__()


class DemandsGenerator:
    def __init__(self, region: SquareRegion, num_demands_pts: int, seed=11):
        self.region = region
        self.num_demands_pts = num_demands_pts
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        x_coords = np.random.uniform(
            low=self.region.x_min, 
            high=self.region.x_max, 
            size=self.num_demands_pts
        )
        y_coords = np.random.uniform(
            low=self.region.y_min, 
            high=self.region.y_max, 
            size=self.num_demands_pts
        )
        demands = np.array([
            Demand(Coordinate(x_coords[k], y_coords[k]), 1) 
            for k in range(self.num_demands_pts)
        ])
        return demands


class Solution:
    def __init__(self, region: SquareRegion, demands, routes):
        self.region = region
        self.demands = demands
        self.routes = routes

    def evaluate(self):
        return 0


class Polyhedron:
    def __init__(self, A, b, B, c, dimension):
        '''
        Polyhedron determined by Ax<=b form and Bx=c
        '''
        self.A, self.b = A.copy(), b.copy()
        self.B, self.c = B.copy(), c.copy()
        self.dim = dimension
        self.update_constraints()

    def update_constraints(self):
        self.eq_constraints = {'type': 'eq', 'fun': lambda x: self.B @ x - self.c , 'jac': lambda _: self.B}
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x, 'jac': lambda _: -self.A}

    def add_ineq_constraint(self, ai, bi):
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.update_constraints()

    def find_analytic_center_with_phase1(self, x0=None):
        """
        Finds the analytic center using a robust scipy optimizer.
        First, it uses a Phase I method (Gurobi) to find a strictly feasible point.
        Then, it minimizes -sum(log(s_i)) where s_i are slacks.
        """
        import sys
        import gurobipy as gp
        from scipy.optimize import minimize

        # --- Phase I: Find a strictly feasible starting point ---
        print("  [Analytic Center] Running Phase I to find a strictly feasible point...")
        n = self.dim
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("phase_I", env=env) as model:
                    x = model.addMVar(shape=n, lb=-np.inf, name="x")
                    t = model.addMVar(shape=1, lb=-np.inf, name="t")
                    model.addConstr(self.A @ x - self.b <= -t)
                    model.addConstr(self.B @ x == self.c)
                    model.setObjective(t, gp.GRB.MAXIMIZE)
                    model.optimize()

                    if model.status != gp.GRB.OPTIMAL or t.X[0] <= 1e-6:
                        print("  [Analytic Center] Warning: Could not find a strictly interior point.", flush=True)
                        if model.status == gp.GRB.OPTIMAL:
                            x0_feasible = x.X
                        else:
                            print("  [Analytic Center] FATAL: Gurobi could not find a feasible point.", flush=True)
                            return np.zeros(self.dim), -np.inf
                    else:
                        x0_feasible = x.X
                        print(f"  [Analytic Center] Phase I successful. Found point with min slack {t.X[0]:.4e}", flush=True)
        except gp.GurobiError as e:
            print(f"  [Analytic Center] Gurobi error in Phase I: {e}", flush=True)
            return np.zeros(self.dim), -np.inf

        # --- Phase II: Find the analytic center using scipy ---
        print("  [Analytic Center] Running Phase II to find analytic center...", flush=True)
        
        def objective(x):
            slacks = self.b - self.A @ x
            return -np.sum(np.log(slacks + 1e-9))

        def objective_jac(x):
            slacks = self.b - self.A @ x
            return (self.A.T / (slacks + 1e-9)).sum(axis=1)

        res = minimize(
            fun=objective,
            x0=x0_feasible,
            method='trust-constr',
            jac=objective_jac,
            constraints=[self.eq_constraints, self.ineq_constraints],
            options={'disp': False, 'maxiter': 200}
        )

        if res.success:
            print("  [Analytic Center] Scipy successfully found the analytic center.", flush=True)
            return res.x, -res.fun
        else:
            print(f"  [Analytic Center] Warning: Scipy failed: {res.message}", flush=True)
            print("  [Analytic Center] Returning last feasible point from Phase I.", flush=True)
            return x0_feasible, -objective(x0_feasible) 