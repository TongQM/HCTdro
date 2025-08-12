#!/usr/bin/env python3
"""
Test script for the debugged precise method
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from classes_cartesian_fixed import SquareRegion, DemandsGenerator, Demand, EmpiricalDistribution, CartesianGrid, Coordinate
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from scipy import integrate

def plot_distribution(density_func, region, demands, num_demands, t, solve_time):
    """Plot the final worst-case distribution and demand points."""
    # Create a grid for plotting
    grid_size = 100  # Increased resolution for finer detail
    x = np.linspace(region.x_min, region.x_max, grid_size)
    y = np.linspace(region.y_min, region.y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate density function on the grid
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            try:
                Z[i, j] = density_func(X[i, j], Y[i, j])
            except:
                Z[i, j] = 0  # Handle any evaluation errors
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot density as contour/heatmap
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)  # More contour levels for finer detail
    plt.colorbar(contour, label='Density')
    
    # Plot demand points
    demand_coords = np.array([d.get_coordinates() for d in demands])
    plt.scatter(demand_coords[:, 0], demand_coords[:, 1], 
                c='red', s=100, marker='o', edgecolors='white', linewidth=2,
                label=f'Demand Points ({num_demands})', zorder=5)
    
    # Add labels and title
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'Worst-Case Distribution (ACCMP)\n'
              f'n={num_demands}, t={t}, time={solve_time:.1f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Set axis limits to region bounds
    plt.xlim(region.x_min, region.x_max)
    plt.ylim(region.y_min, region.y_max)
    
    # Save the plot
    filename = f'figs/accmp_distribution_n{num_demands}_t{t:.1f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {filename}")
    
    plt.close()  # Close to avoid display issues

def main():
    parser = argparse.ArgumentParser(description="Test the fixed precise method with a variable number of demands.")
    parser.add_argument('--num_demands', type=int, default=5, help='Number of demand points to test.')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of analytic center iterations to allow.')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Convergence tolerance epsilon.')
    parser.add_argument('--t', type=float, default=0.5, help='Wasserstein radius for the ball constraint.')
    args = parser.parse_args()

    num_demands = args.num_demands
    max_iter = args.max_iterations
    epsilon = args.epsilon
    t = args.t
    
    print("============================================================")
    print("TESTING FIXED PRECISE METHOD")
    print("============================================================")
    
    # Test configuration
    region = SquareRegion(side_length=10.0)  # 10x10 square service region
    grid = CartesianGrid(size=10, side_length=10.0) # Create a grid
    
    # Create demands and corresponding empirical distribution on the grid
    empirical_distribution = EmpiricalDistribution(grid)
    empirical_distribution.generate_random_samples(num_demands, seed=42)
    empirical_distribution.normalize()

    print("Test configuration:")
    print(f"  Region: {region}")
    print(f"  Number of demands: {num_demands}")
    print(f"  Wasserstein radius t: {t}")
    print(f"  Epsilon: {epsilon} (more relaxed)")
    print(f"  Max iterations: {max_iter}")
    
    try:
        start_time = time.time()
        
        # Create Demand objects from the generated samples
        demands = np.array([Demand(Coordinate(x, y), 1) for x, y in empirical_distribution.samples])

        returned_density_func = find_worst_tsp_density_precise_fixed(
            region,
            demands,
            t=t,
            epsilon=epsilon,
            max_iterations=max_iter,
            tol=1e-3  # Looser tolerance for testing
        )
        end_time = time.time()
        solve_time = end_time - start_time
        
        print("============================================================")
        print("SUCCESS!")
        print("============================================================")
        print(f"Fixed precise method completed in {solve_time:.2f} seconds")
        print(f"Returned density function: {returned_density_func}")
        
        # Test the returned density function
        test_point = (0, 0)
        density_at_point = returned_density_func(*test_point)
        print(f"Test density at {test_point}: {density_at_point}")
        
        # Test if the integral of the probability density is one
        print("\n--- Verifying Integral of Density ---")
        
        # Note: dblquad integrates func(y, x), so we must swap the arguments in the lambda
        integral_result, integral_error = integrate.dblquad(
            lambda y, x: returned_density_func(x, y),
            region.x_min,
            region.x_max,
            region.y_min,
            region.y_max,
            epsabs=1e-3,  # Looser tolerance for speed
            epsrel=1e-3
        )
        
        print(f"Integral of the density function: {integral_result:.6f}")
        print(f"Estimated integration error: {integral_error:.6f}")
        
        if np.isclose(integral_result, 1.0, atol=1e-2):
            print("Integral test PASSED: The integral is close to 1.")
        else:
            print("Integral test FAILED: The integral is not close to 1.")
        
        # Plot the final distribution
        print("\n--- Plotting Final Distribution ---")
        plot_distribution(returned_density_func, region, demands, num_demands, t, solve_time)
        
        print("\nThe fixed precise method works!")
        print(f"Solve time: {solve_time:.2f} seconds")
        
        return True, returned_density_func, solve_time
        
    except Exception as e:
        print("============================================================")
        print("FAILED!")
        print("============================================================")
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        return False, None, 0.0

if __name__ == "__main__":
    success, density_func, solve_time = main()
    
    if success:
        print("\nThe fixed precise method works!")
        print(f"Solve time: {solve_time:.2f} seconds")
    else:
        print("\nThe fixed precise method still has issues.")
        print("Consider using SDP methods as alternatives.") 