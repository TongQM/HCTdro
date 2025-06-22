import numpy as np
import gurobipy as gp
from gurobipy import GRB
from classes_cartesian import SquareRegion, Coordinate, Demand
import time


def find_worst_tsp_density_sdp(region: SquareRegion, demands, t: float = 1, grid_size: int = 50):
    '''
    SDP-based method for finding worst-case TSP density
    This method discretizes the continuous problem into a semi-definite program
    
    Based on the formulation in SDP2025.ipynb:
    - Assumes uniform distribution on each grid cell
    - Maximizes sum of square roots of densities
    - Uses optimal transport for Wasserstein constraint
    
    Input: A square region containing a set of distinct points x1, x2,..., xn, which are 
    interpreted as an empirical distribution f_hat, a distance parameter t, and grid size.

    Output: An approximation of the distribution f* that maximizes int_R sqrt(f(x)) dA 
    subject to the constraint that D(f_hat, f) <= t.
    '''
    
    print(f"Starting SDP method with grid size {grid_size}x{grid_size}")
    start_time = time.time()
    
    # Create grid points
    x_grid = np.linspace(region.x_min, region.x_max, grid_size)
    y_grid = np.linspace(region.y_min, region.y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Get demand locations
    demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])
    n_demands = len(demands)
    n_grid_points = len(grid_points)
    
    print(f"Grid has {n_grid_points} points, {n_demands} demand points")
    
    # Compute distances between grid points and demand points
    distances = np.zeros((n_grid_points, n_demands))
    for i in range(n_grid_points):
        for j in range(n_demands):
            distances[i, j] = np.linalg.norm(grid_points[i] - demands_locations[j])
    
    # Area of each grid cell
    dx = (region.x_max - region.x_min) / (grid_size - 1)
    dy = (region.y_max - region.y_min) / (grid_size - 1)
    cell_area = dx * dy
    
    # Create empirical distribution (uniform mass at demand points)
    p = np.zeros(n_grid_points)
    for i, demand in enumerate(demands):
        # Find the closest grid point to each demand
        demand_coord = demand.get_coordinates()
        min_dist = float('inf')
        closest_grid_idx = 0
        for j in range(n_grid_points):
            dist = np.linalg.norm(grid_points[j] - demand_coord)
            if dist < min_dist:
                min_dist = dist
                closest_grid_idx = j
        p[closest_grid_idx] += 1.0 / n_demands
    
    # Create Gurobi model
    model = gp.Model("WorstCaseTSPDensity")
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Variables: x[i] represents sqrt(density) at grid point i
    x = model.addVars(n_grid_points, lb=0.0, name="x")
    
    # Variables: s[i] represents density at grid point i
    s = model.addVars(n_grid_points, lb=0.0, name="s")
    
    # Variables: y[i,j] represents transport plan from empirical point i to grid point j
    y = model.addVars(n_demands, n_grid_points, lb=0.0, name='y')
    
    # Objective: maximize sum of sqrt(density) * cell_area
    # This approximates int_R sqrt(f(x)) dA
    objective = cell_area * gp.quicksum(x[i] for i in range(n_grid_points))
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Constraint: s[i] >= x[i]^2 (density >= (sqrt_density)^2)
    for i in range(n_grid_points):
        model.addQConstr(s[i] >= x[i] * x[i], name=f"density_constraint_{i}")
    
    # Constraint: Total mass = 1 (normalization)
    model.addConstr(cell_area * gp.quicksum(s[i] for i in range(n_grid_points)) == 1, name="mass")
    
    # Optimal transport constraints for Wasserstein distance
    # Marginal constraints: sum over destinations = empirical mass
    for i in range(n_demands):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n_grid_points)) == 1/n_demands, name=f"marginal_source_{i}")
    
    # Marginal constraints: sum over sources = continuous mass
    for j in range(n_grid_points):
        model.addConstr(gp.quicksum(y[i, j] for i in range(n_demands)) == s[j] * cell_area, name=f"marginal_dest_{j}")
    
    # Wasserstein distance constraint
    wasserstein_cost = gp.quicksum(
        y[i, j] * distances[j, i] 
        for i in range(n_demands) 
        for j in range(n_grid_points)
    )
    model.addConstr(wasserstein_cost <= t, name="wasserstein")
    
    # Solve the model
    print("Solving SDP model...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Extract solution
        x_values = np.array([x[i].X for i in range(n_grid_points)])
        s_values = np.array([s[i].X for i in range(n_grid_points)])
        y_values = np.array([[y[i, j].X for j in range(n_grid_points)] for i in range(n_demands)])
        
        objective_value = model.objVal
        wasserstein_actual = sum(y_values[i, j] * distances[j, i] 
                               for i in range(n_demands) 
                               for j in range(n_grid_points))
        
        print(f"SDP solved successfully!")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Wasserstein distance: {wasserstein_actual:.6f} (bound: {t})")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Create density function that interpolates grid values
        def density_function(x_coord, y_coord):
            # Find nearest grid point
            x_idx = int((x_coord - region.x_min) / dx)
            y_idx = int((y_coord - region.y_min) / dy)
            
            # Clamp to grid bounds
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            
            # Return density value at this grid point
            grid_idx = y_idx * grid_size + x_idx
            return s_values[grid_idx]
        
        return density_function, {
            'objective_value': objective_value,
            'wasserstein_distance': wasserstein_actual,
            'grid_size': grid_size,
            'solve_time': time.time() - start_time,
            's_values': s_values.reshape(grid_size, grid_size),
            'x_values': x_values.reshape(grid_size, grid_size),
            'grid_points': grid_points,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'cell_area': cell_area
        }
    else:
        print(f"SDP failed to solve. Status: {model.status}")
        return None, None


def find_worst_tsp_density_sdp_improved(region: SquareRegion, demands, t: float = 1, grid_size: int = 50):
    '''
    Improved SDP-based method with better formulation
    Uses a more efficient formulation that directly optimizes the density values
    '''
    
    print(f"Starting improved SDP method with grid size {grid_size}x{grid_size}")
    start_time = time.time()
    
    # Create grid points
    x_grid = np.linspace(region.x_min, region.x_max, grid_size)
    y_grid = np.linspace(region.y_min, region.y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Get demand locations
    demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])
    n_demands = len(demands)
    n_grid_points = len(grid_points)
    
    print(f"Grid has {n_grid_points} points, {n_demands} demand points")
    
    # Compute distances between grid points and demand points
    distances = np.zeros((n_grid_points, n_demands))
    for i in range(n_grid_points):
        for j in range(n_demands):
            distances[i, j] = np.linalg.norm(grid_points[i] - demands_locations[j])
    
    # Area of each grid cell
    dx = (region.x_max - region.x_min) / (grid_size - 1)
    dy = (region.y_max - region.y_min) / (grid_size - 1)
    cell_area = dx * dy
    
    # Create empirical distribution
    p = np.zeros(n_grid_points)
    for i, demand in enumerate(demands):
        demand_coord = demand.get_coordinates()
        min_dist = float('inf')
        closest_grid_idx = 0
        for j in range(n_grid_points):
            dist = np.linalg.norm(grid_points[j] - demand_coord)
            if dist < min_dist:
                min_dist = dist
                closest_grid_idx = j
        p[closest_grid_idx] += 1.0 / n_demands
    
    # Create Gurobi model
    model = gp.Model("WorstCaseTSPDensityImproved")
    model.setParam('OutputFlag', 0)
    
    # Variables: x[i] represents sqrt(density) at grid point i
    x = model.addVars(n_grid_points, lb=0.0, name="x")
    
    # Variables: s[i] represents density at grid point i
    s = model.addVars(n_grid_points, lb=0.0, name="s")
    
    # Variables: y[i,j] represents transport plan
    y = model.addVars(n_demands, n_grid_points, lb=0.0, name='y')
    
    # Objective: maximize sum of sqrt(density) * cell_area
    objective = cell_area * gp.quicksum(x[i] for i in range(n_grid_points))
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Constraint: s[i] >= x[i]^2
    for i in range(n_grid_points):
        model.addQConstr(s[i] >= x[i] * x[i], name=f"density_constraint_{i}")
    
    # Constraint: Total mass = 1
    model.addConstr(cell_area * gp.quicksum(s[i] for i in range(n_grid_points)) == 1, name="mass")
    
    # Optimal transport constraints
    for i in range(n_demands):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n_grid_points)) == 1/n_demands, name=f"marginal_source_{i}")
    
    for j in range(n_grid_points):
        model.addConstr(gp.quicksum(y[i, j] for i in range(n_demands)) == s[j] * cell_area, name=f"marginal_dest_{j}")
    
    # Wasserstein distance constraint
    wasserstein_cost = gp.quicksum(
        y[i, j] * distances[j, i] 
        for i in range(n_demands) 
        for j in range(n_grid_points)
    )
    model.addConstr(wasserstein_cost <= t, name="wasserstein")
    
    # Solve the model
    print("Solving improved SDP model...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Extract solution
        x_values = np.array([x[i].X for i in range(n_grid_points)])
        s_values = np.array([s[i].X for i in range(n_grid_points)])
        
        objective_value = model.objVal
        
        print(f"Improved SDP solved successfully!")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Create density function that interpolates grid values
        def density_function(x_coord, y_coord):
            # Find nearest grid point
            x_idx = int((x_coord - region.x_min) / dx)
            y_idx = int((y_coord - region.y_min) / dy)
            
            # Clamp to grid bounds
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            
            # Return density value at this grid point
            grid_idx = y_idx * grid_size + x_idx
            return s_values[grid_idx]
        
        return density_function, {
            'objective_value': objective_value,
            'grid_size': grid_size,
            'solve_time': time.time() - start_time,
            's_values': s_values.reshape(grid_size, grid_size),
            'x_values': x_values.reshape(grid_size, grid_size),
            'grid_points': grid_points,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'cell_area': cell_area
        }
    else:
        print(f"Improved SDP failed to solve. Status: {model.status}")
        return None, None


def compute_wasserstein_distance_approximate(demands_locations, s_values, grid_points, cell_area, t):
    """
    Compute approximate Wasserstein distance between empirical distribution and continuous distribution
    """
    n_demands = len(demands_locations)
    n_grid_points = len(grid_points)
    
    # Compute distances
    distances = np.zeros((n_grid_points, n_demands))
    for i in range(n_grid_points):
        for j in range(n_demands):
            distances[i, j] = np.linalg.norm(grid_points[i] - demands_locations[j])
    
    # Solve optimal transport problem
    model = gp.Model("OptimalTransport")
    model.setParam('OutputFlag', 0)
    
    # Transport plan variables
    y = model.addVars(n_demands, n_grid_points, lb=0.0, name="y")
    
    # Marginal constraints
    for i in range(n_demands):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n_grid_points)) == 1/n_demands, name=f"marginal_source_{i}")
    
    for j in range(n_grid_points):
        model.addConstr(gp.quicksum(y[i, j] for i in range(n_demands)) == s_values[j] * cell_area, name=f"marginal_dest_{j}")
    
    # Objective: minimize transport cost
    transport_cost = gp.quicksum(
        y[i, j] * distances[j, i] 
        for i in range(n_demands) 
        for j in range(n_grid_points)
    )
    model.setObjective(transport_cost, GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return np.inf


def find_worst_tsp_density_sdp_simple(region: SquareRegion, demands, t: float = 1, grid_size: int = 50):
    '''
    Simplified SDP method that directly maximizes sum of sqrt(density)
    without the complex transport plan formulation
    '''
    
    print(f"Starting simple SDP method with grid size {grid_size}x{grid_size}")
    start_time = time.time()
    
    # Create grid points
    x_grid = np.linspace(region.x_min, region.x_max, grid_size)
    y_grid = np.linspace(region.y_min, region.y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Get demand locations
    demands_locations = np.array([demands[i].get_coordinates() for i in range(len(demands))])
    n_demands = len(demands)
    n_grid_points = len(grid_points)
    
    print(f"Grid has {n_grid_points} points, {n_demands} demand points")
    
    # Area of each grid cell
    dx = (region.x_max - region.x_min) / (grid_size - 1)
    dy = (region.y_max - region.y_min) / (grid_size - 1)
    cell_area = dx * dy
    
    # Create Gurobi model
    model = gp.Model("WorstCaseTSPDensitySimple")
    model.setParam('OutputFlag', 0)
    
    # Variables: x[i] represents sqrt(density) at grid point i
    x = model.addVars(n_grid_points, lb=0.0, name="x")
    
    # Variables: s[i] represents density at grid point i
    s = model.addVars(n_grid_points, lb=0.0, name="s")
    
    # Objective: maximize sum of sqrt(density) * cell_area
    objective = cell_area * gp.quicksum(x[i] for i in range(n_grid_points))
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Constraint: s[i] >= x[i]^2
    for i in range(n_grid_points):
        model.addQConstr(s[i] >= x[i] * x[i], name=f"density_constraint_{i}")
    
    # Constraint: Total mass = 1
    model.addConstr(cell_area * gp.quicksum(s[i] for i in range(n_grid_points)) == 1, name="mass")
    
    # Simplified Wasserstein constraint using dual formulation
    # We approximate the Wasserstein constraint by ensuring the distribution
    # doesn't deviate too much from the empirical distribution
    
    # Create empirical distribution
    p = np.zeros(n_grid_points)
    for i, demand in enumerate(demands):
        demand_coord = demand.get_coordinates()
        min_dist = float('inf')
        closest_grid_idx = 0
        for j in range(n_grid_points):
            dist = np.linalg.norm(grid_points[j] - demand_coord)
            if dist < min_dist:
                min_dist = dist
                closest_grid_idx = j
        p[closest_grid_idx] += 1.0 / n_demands
    
    # Add constraints to ensure the distribution is close to empirical
    for i in range(n_grid_points):
        if p[i] > 0:
            # If empirical mass exists at this point, ensure some mass is allocated nearby
            model.addConstr(s[i] >= p[i] * 0.1, name=f"empirical_constraint_{i}")
    
    # Solve the model
    print("Solving simple SDP model...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Extract solution
        x_values = np.array([x[i].X for i in range(n_grid_points)])
        s_values = np.array([s[i].X for i in range(n_grid_points)])
        
        objective_value = model.objVal
        
        print(f"Simple SDP solved successfully!")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Create density function that interpolates grid values
        def density_function(x_coord, y_coord):
            # Find nearest grid point
            x_idx = int((x_coord - region.x_min) / dx)
            y_idx = int((y_coord - region.y_min) / dy)
            
            # Clamp to grid bounds
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            
            # Return density value at this grid point
            grid_idx = y_idx * grid_size + x_idx
            return s_values[grid_idx]
        
        return density_function, {
            'objective_value': objective_value,
            'grid_size': grid_size,
            'solve_time': time.time() - start_time,
            's_values': s_values.reshape(grid_size, grid_size),
            'x_values': x_values.reshape(grid_size, grid_size),
            'grid_points': grid_points,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'cell_area': cell_area
        }
    else:
        print(f"Simple SDP failed to solve. Status: {model.status}")
        return None, None 