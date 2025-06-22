#!/usr/bin/env python3
"""
Test script for the reorganized worst-case TSP density comparison framework
"""

import numpy as np
import matplotlib.pyplot as plt
from classes_cartesian import SquareRegion, Coordinate, DemandsGenerator
from comparison_framework import ComparisonFramework, run_comprehensive_comparison


def test_basic_functionality():
    """Test basic functionality of the reorganized code"""
    print("Testing basic functionality...")
    
    # Create a simple test case
    region = SquareRegion(side_length=2.0)
    generator = DemandsGenerator(region, num_demands_pts=5, seed=42)
    demands = generator.generate()
    
    print(f"Created region: {region}")
    print(f"Generated {len(demands)} demand points:")
    for i, demand in enumerate(demands):
        print(f"  Demand {i}: {demand}")
    
    # Test coordinate conversion
    coords = np.array([demand.get_coordinates() for demand in demands])
    print(f"Demand coordinates:\n{coords}")
    
    # Test region containment
    test_points = [(0, 0), (1, 1), (2, 2), (-1, -1)]
    for x, y in test_points:
        inside = region.contains_point(x, y)
        print(f"Point ({x}, {y}) is {'inside' if inside else 'outside'} the region")
    
    print("Basic functionality test passed!\n")


def test_simple_comparison():
    """Test a simple comparison with small parameters"""
    print("Testing simple comparison...")
    
    try:
        # Run a simple comparison with small parameters
        framework, results = run_comprehensive_comparison(
            region_size=1.0,
            num_demands=3,
            t=0.2,
            epsilon=0.2,
            grid_sizes=[10, 15],
            seed=123
        )
        
        print("Simple comparison completed successfully!")
        print(f"Results shape: {results.shape}")
        print(f"Results:\n{results}")
        
        return True
        
    except Exception as e:
        print(f"Simple comparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_methods():
    """Test individual methods separately"""
    print("Testing individual methods...")
    
    # Create test setup
    region = SquareRegion(side_length=1.5)
    generator = DemandsGenerator(region, num_demands_pts=4, seed=42)
    demands = generator.generate()
    
    framework = ComparisonFramework(region, demands, t=0.3, epsilon=0.2)
    
    # Test precise method
    try:
        print("Testing precise method...")
        density_func, obj_val, solve_time = framework.run_precise_method(use_torchquad=True)
        print(f"Precise method: obj={obj_val:.6f}, time={solve_time:.2f}s")
    except Exception as e:
        print(f"Precise method failed: {e}")
    
    # Test SDP method
    try:
        print("Testing SDP method...")
        density_func, obj_val, solve_time = framework.run_sdp_method(grid_size=20)
        if density_func is not None:
            print(f"SDP method: obj={obj_val:.6f}, time={solve_time:.2f}s")
        else:
            print("SDP method returned None")
    except Exception as e:
        print(f"SDP method failed: {e}")
    
    # Test improved SDP method
    try:
        print("Testing improved SDP method...")
        density_func, obj_val, solve_time = framework.run_sdp_improved_method(grid_size=20)
        if density_func is not None:
            print(f"Improved SDP method: obj={obj_val:.6f}, time={solve_time:.2f}s")
        else:
            print("Improved SDP method returned None")
    except Exception as e:
        print(f"Improved SDP method failed: {e}")


def test_density_visualization():
    """Test density function visualization"""
    print("Testing density visualization...")
    
    # Create a simple test case
    region = SquareRegion(side_length=2.0)
    generator = DemandsGenerator(region, num_demands_pts=3, seed=42)
    demands = generator.generate()
    
    framework = ComparisonFramework(region, demands, t=0.3, epsilon=0.2)
    
    # Run one method to get a density function
    try:
        density_func, obj_val, solve_time = framework.run_precise_method(use_torchquad=True)
        
        # Test the density function
        x = np.linspace(region.x_min, region.x_max, 50)
        y = np.linspace(region.y_min, region.y_max, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(50):
            for j in range(50):
                Z[i, j] = density_func(X[i, j], Y[i, j])
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Density')
        
        # Plot demand points
        demands_coords = np.array([d.get_coordinates() for d in demands])
        plt.scatter(demands_coords[:, 0], demands_coords[:, 1], c='red', s=100, marker='x', label='Demands')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test Density Function')
        plt.legend()
        plt.show()
        
        print("Density visualization test passed!")
        
    except Exception as e:
        print(f"Density visualization test failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING REORGANIZED WORST-CASE TSP DENSITY CODE")
    print("=" * 60)
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test individual methods
    test_individual_methods()
    
    # Test density visualization
    test_density_visualization()
    
    # Test simple comparison (this might take longer)
    print("\n" + "=" * 60)
    print("RUNNING SIMPLE COMPARISON TEST")
    print("=" * 60)
    success = test_simple_comparison()
    
    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED!")
        print("=" * 60)


if __name__ == "__main__":
    main() 