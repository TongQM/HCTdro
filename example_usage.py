#!/usr/bin/env python3
"""
Example usage of the reorganized worst-case TSP density comparison framework
"""

import numpy as np
import matplotlib.pyplot as plt
from classes_cartesian import SquareRegion, Coordinate, DemandsGenerator
from precise_method_cartesian import find_worst_tsp_density_precise
from sdp_method_cartesian import find_worst_tsp_density_sdp
from comparison_framework import ComparisonFramework


def example_basic_usage():
    """Basic example showing how to use individual methods"""
    print("=" * 50)
    print("BASIC USAGE EXAMPLE")
    print("=" * 50)
    
    # Create a square region
    region = SquareRegion(side_length=2.0, center_x=0.0, center_y=0.0)
    print(f"Created region: {region}")
    
    # Generate random demand points
    generator = DemandsGenerator(region, num_demands_pts=5, seed=42)
    demands = generator.generate()
    print(f"Generated {len(demands)} demand points")
    
    # Show demand coordinates
    for i, demand in enumerate(demands):
        print(f"  Demand {i}: {demand}")
    
    # Parameters
    t = 0.3  # Wasserstein distance bound
    epsilon = 0.1  # Tolerance for precise method
    
    print(f"\nParameters: t={t}, epsilon={epsilon}")
    
    return region, demands, t, epsilon


def example_precise_method(region, demands, t, epsilon):
    """Example of using the precise method"""
    print("\n" + "=" * 50)
    print("PRECISE METHOD EXAMPLE")
    print("=" * 50)
    
    try:
        # Run precise method with torchquad (faster)
        print("Running precise method with torchquad...")
        density_func = find_worst_tsp_density_precise(
            region, demands, t, epsilon, use_torchquad=True
        )
        
        # Test the density function at a few points
        test_points = [(0, 0), (0.5, 0.5), (-0.5, -0.5)]
        print("\nDensity function values at test points:")
        for x, y in test_points:
            if region.contains_point(x, y):
                density = density_func(x, y)
                print(f"  f({x}, {y}) = {density:.6f}")
            else:
                print(f"  f({x}, {y}) = N/A (outside region)")
        
        return density_func
        
    except Exception as e:
        print(f"Precise method failed: {e}")
        return None


def example_sdp_method(region, demands, t):
    """Example of using the SDP method"""
    print("\n" + "=" * 50)
    print("SDP METHOD EXAMPLE")
    print("=" * 50)
    
    try:
        # Run SDP method with different grid sizes
        grid_sizes = [20, 30]
        
        for grid_size in grid_sizes:
            print(f"\nRunning SDP method with grid size {grid_size}...")
            density_func, sdp_info = find_worst_tsp_density_sdp(
                region, demands, t, grid_size
            )
            
            if density_func is not None:
                print(f"  Objective value: {sdp_info['objective_value']:.6f}")
                print(f"  Solve time: {sdp_info['solve_time']:.2f} seconds")
                print(f"  Grid size: {sdp_info['grid_size']}")
                
                # Test the density function
                test_point = (0, 0)
                if region.contains_point(*test_point):
                    density = density_func(*test_point)
                    print(f"  f{test_point} = {density:.6f}")
        
        return density_func, sdp_info
        
    except Exception as e:
        print(f"SDP method failed: {e}")
        return None, None


def example_comparison_framework(region, demands, t, epsilon):
    """Example of using the comparison framework"""
    print("\n" + "=" * 50)
    print("COMPARISON FRAMEWORK EXAMPLE")
    print("=" * 50)
    
    try:
        # Create comparison framework
        framework = ComparisonFramework(region, demands, t, epsilon)
        
        # Run comparison with smaller grid sizes for demonstration
        df = framework.compare_methods(
            methods_to_run=['precise', 'sdp'],
            grid_sizes=[15, 20],
            use_torchquad=True
        )
        
        print("\nComparison Results:")
        print(df.to_string(index=False))
        
        # Plot density comparison (if matplotlib is available)
        try:
            framework.plot_density_comparison(resolution=50)
            print("\nDensity comparison plot displayed.")
        except Exception as e:
            print(f"Could not display plot: {e}")
        
        return framework, df
        
    except Exception as e:
        print(f"Comparison framework failed: {e}")
        return None, None


def example_visualization(region, demands, density_func):
    """Example of visualizing the density function"""
    print("\n" + "=" * 50)
    print("VISUALIZATION EXAMPLE")
    print("=" * 50)
    
    if density_func is None:
        print("No density function to visualize.")
        return
    
    try:
        # Create grid for visualization
        resolution = 50
        x = np.linspace(region.x_min, region.x_max, resolution)
        y = np.linspace(region.y_min, region.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Compute density values
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = density_func(X[i, j], Y[i, j])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot density
        im = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(im, label='Density')
        
        # Plot demand points
        demands_coords = np.array([d.get_coordinates() for d in demands])
        plt.scatter(demands_coords[:, 0], demands_coords[:, 1], 
                   c='red', s=100, marker='x', label='Demand Points')
        
        # Plot region boundary
        plt.plot([region.x_min, region.x_max, region.x_max, region.x_min, region.x_min],
                [region.y_min, region.y_min, region.y_max, region.y_max, region.y_min],
                'k--', linewidth=2, label='Region Boundary')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Worst-Case TSP Density Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print("Density visualization completed.")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Run all examples"""
    print("WORST-CASE TSP DENSITY COMPARISON FRAMEWORK")
    print("Example Usage")
    print("=" * 60)
    
    # Basic setup
    region, demands, t, epsilon = example_basic_usage()
    
    # Precise method example
    density_func_precise = example_precise_method(region, demands, t, epsilon)
    
    # SDP method example
    density_func_sdp, sdp_info = example_sdp_method(region, demands, t)
    
    # Comparison framework example
    framework, df = example_comparison_framework(region, demands, t, epsilon)
    
    # Visualization example
    if density_func_precise is not None:
        example_visualization(region, demands, density_func_precise)
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)
    print("This demonstrates the basic usage of the reorganized framework.")
    print("For more advanced usage, see the README and test files.")


if __name__ == "__main__":
    main() 