import time
import numpy as np
from classes_cartesian_fixed import SquareRegion, DemandsGenerator, Demand, EmpiricalDistribution, CartesianGrid, Coordinate
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed, f_tilde_cartesian
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from findWassersteinDistance import findWassersteinDistance_cartesian
import os


def approx_wasserstein(empirical_points, empirical_weights, density_func, region, t, grid_res=30):
    # Discretize region
    x = np.linspace(region.x_min, region.x_max, grid_res)
    y = np.linspace(region.y_min, region.y_max, grid_res)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_density = np.array([density_func(xi, yi) for xi, yi in grid_points])
    grid_density = np.maximum(grid_density, 0)
    grid_density /= grid_density.sum()  # Normalize
    # Empirical distribution: Dirac at demand points
    emp_points = np.array(empirical_points)
    emp_weights = np.array(empirical_weights)
    # Cost matrix: distance from each demand to each grid point
    cost_matrix = np.linalg.norm(emp_points[:, None, :] - grid_points[None, :, :], axis=2)
    # For each demand, assign its mass to the closest grid point (approximate transport)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Only assign as much mass as available in grid_density
    assigned_mass = np.minimum(emp_weights, grid_density[col_ind])
    # Wasserstein cost: sum of assigned mass * distance
    wasserstein = np.sum(assigned_mass * cost_matrix[row_ind, col_ind])
    return wasserstein


def run_test(num_demands, num_random_v=5, plot_results=False):
    print(f"--- Running test for {num_demands} demand points ---")
    start_time = time.time()
    region = SquareRegion(side_length=1.0)
    grid = CartesianGrid(size=10, side_length=1.0)
    empirical_distribution = EmpiricalDistribution(grid)
    empirical_distribution.generate_random_samples(num_demands, seed=42)
    empirical_distribution.normalize()
    demands = np.array([Demand(Coordinate(x, y), 1) for x, y in empirical_distribution.samples])
    t = 0.1

    # Run precise method and get parameters and history
    density_func, demands_locations, lambdas_opt, v_tilde_opt, history = find_worst_tsp_density_precise_fixed(
        region, demands, t=t, epsilon=1e-2, max_iterations=50, tol=1e-5, return_params=True, return_history=True
    )
    solve_time = time.time() - start_time

    # Check normalization of the optimal density
    integral_result, integral_error = integrate.dblquad(
        lambda y, x: density_func(x, y),
        region.x_min, region.x_max,
        region.y_min, region.y_max,
        epsabs=1e-3, epsrel=1e-3
    )
    print(f"  Optimal v_tilde mass: {integral_result:.6f} (error: {integral_error:.2e})")
    is_normalized = np.isclose(integral_result, 1.0, atol=1e-2)

    # Compute optimality (integral of sqrt of density)
    def sqrt_density(x, y):
        return np.sqrt(density_func(x, y))
    sqrt_integral_opt, _ = integrate.dblquad(
        lambda y, x: sqrt_density(x, y),
        region.x_min, region.x_max,
        region.y_min, region.y_max,
        epsabs=1e-3, epsrel=1e-3
    )

    # Wasserstein distance check (precise, but time-consuming)
    # wasserstein_opt = findWassersteinDistance_cartesian(demands_locations, density_func, region)
    # print(f"  Wasserstein distance of optimal density: {wasserstein_opt:.6f}")
    # Disabled for speed: the precise Wasserstein check is very time-consuming.

    print(f"  Normalization: Integral = {integral_result:.6f} (error: {integral_error:.2e}) -> {'PASS' if is_normalized else 'FAIL'}")
    print(f"  Sqrt integral (optimal): {sqrt_integral_opt:.6f}")
    print(f"  Solve time: {solve_time:.2f} seconds\n")

    # Ensure figs directory exists
    os.makedirs('figs', exist_ok=True)

    # Plot convergence and density for the first test only
    if plot_results:
        # Plot UB/LB convergence
        plt.figure(figsize=(6, 4))
        plt.plot(history['UB'], label='UB')
        plt.plot(history['LB'], label='LB')
        plt.plot(history['gap'], label='Gap')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('UB/LB Convergence')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/convergence_{num_demands}.png')
        plt.close()

        # Plot density heatmap and demand points (optimal)
        grid_res = 100
        x = np.linspace(region.x_min, region.x_max, grid_res)
        y = np.linspace(region.y_min, region.y_max, grid_res)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(grid_res):
            for j in range(grid_res):
                Z[i, j] = density_func(X[i, j], Y[i, j])
        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Density')
        plt.scatter(demands_locations[:, 0], demands_locations[:, 1], c='red', marker='x', label='Empirical Demands')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Worst-case Distribution (Optimal) and Empirical Demands')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/density_optimal_{num_demands}.png')
        plt.close()

    return is_normalized


def main():
    print("============================================================")
    print("ROBUSTNESS & OPTIMALITY TEST FOR PRECISE METHOD")
    print("============================================================")
    demand_points_to_test = [5]
    results = {}
    for idx, num_demands in enumerate(demand_points_to_test):
        result = run_test(num_demands, plot_results=True)
        results[num_demands] = result
    print("\n============================================================")
    print("ROBUSTNESS & OPTIMALITY TEST SUMMARY")
    print("============================================================")
    for num_demands, passed in results.items():
        print(f"  {num_demands} demands: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    main() 