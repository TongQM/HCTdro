# Worst-Case TSP Density Comparison Framework

This repository contains a reorganized implementation for finding worst-case distributions within a Wasserstein ball, comparing the precise method (analytic center cutting plane) with SDP-based approximations. The code has been converted from polar coordinates on a disk to Cartesian coordinates on a square region.

## Overview

The problem is to find the worst-case distribution that maximizes the TSP density objective `∫_R √f(x) dA` subject to a Wasserstein distance constraint `D(f_hat, f) ≤ t`, where `f_hat` is an empirical distribution.

## Key Changes from Original Code

1. **Coordinate System**: Converted from polar coordinates on a disk to Cartesian coordinates on a square region
2. **Method Comparison**: Added SDP-based methods alongside the original precise method
3. **Modular Design**: Separated concerns into different modules for better maintainability
4. **Comprehensive Testing**: Added testing framework and visualization tools
5. **Corrected SDP Formulation**: Updated SDP methods to match the formulation in SDP2025.ipynb

## File Structure

```
├── classes_cartesian.py          # Cartesian coordinate system classes
├── precise_method_cartesian.py   # Original precise method implementation
├── sdp_method_cartesian.py       # SDP-based approximation methods (corrected)
├── comparison_framework.py       # Framework for comparing methods
├── test_comparison.py           # Test script
├── example_usage.py             # Usage examples
└── README_reorganized.md        # This file
```

## Classes and Data Structures

### `classes_cartesian.py`

- **`Coordinate`**: Represents points in Cartesian coordinates (x, y)
- **`SquareRegion`**: Represents a square region with configurable side length and center
- **`Demand`**: Represents a demand point with location and demand value
- **`DemandsGenerator`**: Generates random demand points within a region
- **`Polyhedron`**: Handles polyhedral constraints for the analytic center method

### `precise_method_cartesian.py`

Contains the original precise method implementation using analytic center cutting plane:

- **`find_worst_tsp_density_precise()`**: Main function implementing the precise method
- **`minimize_problem14_cartesian()`**: Solves the upper bounding problem
- **`minimize_problem7_cartesian()`**: Solves the lower bounding problem
- Support for both scipy.integrate and torchquad for numerical integration

### `sdp_method_cartesian.py`

Contains SDP-based approximation methods based on the formulation in SDP2025.ipynb:

- **`find_worst_tsp_density_sdp()`**: Main SDP formulation using optimal transport
- **`find_worst_tsp_density_sdp_improved()`**: Improved SDP formulation
- **`find_worst_tsp_density_sdp_simple()`**: Simplified SDP formulation for faster computation
- **`compute_wasserstein_distance_approximate()`**: Computes approximate Wasserstein distances

**Key SDP Formulation Details:**
- Assumes uniform distribution on each grid cell
- Maximizes `∑ᵢ xᵢ` where `xᵢ` represents `√(density)` at grid point i
- Uses constraint `sᵢ ≥ xᵢ²` to represent density values
- Implements Wasserstein constraint via optimal transport formulation
- Uses transport plan variables `y[i,j]` for empirical to continuous mass transfer

### `comparison_framework.py`

Provides a comprehensive framework for comparing methods:

- **`ComparisonFramework`**: Main class for running and comparing methods
- **`run_comprehensive_comparison()`**: Convenience function for running full comparisons
- Visualization and analysis tools

## Installation and Dependencies

Required packages:
```bash
pip install numpy scipy matplotlib pandas seaborn gurobipy torch torchquad numba
```

Note: Gurobi requires a license. For academic use, you can obtain a free academic license.

## Usage Examples

### Basic Usage

```python
from classes_cartesian import SquareRegion, DemandsGenerator
from comparison_framework import run_comprehensive_comparison

# Run a comprehensive comparison
framework, results = run_comprehensive_comparison(
    region_size=2.0,      # Side length of square region
    num_demands=8,        # Number of demand points
    t=0.3,               # Wasserstein distance bound
    epsilon=0.1,         # Tolerance for precise method
    grid_sizes=[20, 30, 50],  # Grid sizes for SDP methods
    seed=42              # Random seed for reproducibility
)
```

### Individual Method Usage

```python
from classes_cartesian import SquareRegion, DemandsGenerator
from precise_method_cartesian import find_worst_tsp_density_precise
from sdp_method_cartesian import find_worst_tsp_density_sdp

# Create region and demands
region = SquareRegion(side_length=2.0)
generator = DemandsGenerator(region, num_demands_pts=5, seed=42)
demands = generator.generate()

# Run precise method
density_func_precise = find_worst_tsp_density_precise(
    region, demands, t=0.3, epsilon=0.1, use_torchquad=True
)

# Run SDP method (corrected formulation)
density_func_sdp, sdp_info = find_worst_tsp_density_sdp(
    region, demands, t=0.3, grid_size=50
)
```

### Using the Comparison Framework

```python
from comparison_framework import ComparisonFramework

# Create framework
framework = ComparisonFramework(region, demands, t=0.3, epsilon=0.1)

# Run individual methods
framework.run_precise_method(use_torchquad=True)
framework.run_sdp_method(grid_size=50)
framework.run_sdp_improved_method(grid_size=50)
framework.run_sdp_simple_method(grid_size=50)

# Compare all methods
df = framework.compare_methods(
    methods_to_run=['precise', 'sdp', 'sdp_improved', 'sdp_simple'],
    grid_sizes=[30, 50, 100]
)

# Visualize results
framework.plot_density_comparison()
framework.plot_efficiency_comparison(df)

# Save results
framework.save_results('my_results.csv')
```

## Method Comparison

### Precise Method (Analytic Center Cutting Plane)
- **Pros**: Exact solution, theoretically sound
- **Cons**: Computationally expensive, requires multiple iterations
- **Use Case**: When high accuracy is required and computational time is not a constraint

### SDP Method (Optimal Transport)
- **Pros**: Correct formulation based on SDP2025.ipynb, handles Wasserstein constraints properly
- **Cons**: Computationally intensive due to transport plan variables
- **Use Case**: When accurate Wasserstein distance handling is required

### Improved SDP Method (Optimal Transport)
- **Pros**: Same formulation as main SDP but with potential optimizations
- **Cons**: Still computationally intensive
- **Use Case**: When seeking the most accurate SDP approximation

### Simple SDP Method (Approximate)
- **Pros**: Faster computation, simpler formulation
- **Cons**: Approximate Wasserstein constraint handling
- **Use Case**: When speed is important and some approximation error is acceptable

## SDP Formulation Details

The SDP formulation follows the structure from SDP2025.ipynb:

1. **Variables**:
   - `x[i]`: Square root of density at grid point i
   - `s[i]`: Density at grid point i (s[i] ≥ x[i]²)
   - `y[i,j]`: Transport plan from empirical point i to grid point j

2. **Objective**: Maximize `∑ᵢ xᵢ * cell_area` (approximates `∫_R √f(x) dA`)

3. **Constraints**:
   - `s[i] ≥ x[i]²` (density constraint)
   - `∑ᵢ s[i] * cell_area = 1` (mass normalization)
   - `∑ⱼ y[i,j] = 1/n_demands` (empirical marginal)
   - `∑ᵢ y[i,j] = s[j] * cell_area` (continuous marginal)
   - `∑ᵢⱼ y[i,j] * distance[i,j] ≤ t` (Wasserstein constraint)

## Testing

Run the test script to verify the implementation:

```bash
python test_comparison.py
```

This will test:
- Basic functionality of classes and data structures
- Individual method implementations
- Density function visualization
- Simple comparison framework

## Performance Considerations

1. **Grid Size**: Larger grid sizes in SDP methods provide better accuracy but increase computation time
2. **Integration Method**: torchquad is generally faster than scipy.integrate for high-dimensional integration
3. **Number of Demands**: More demand points increase problem complexity for all methods
4. **Wasserstein Bound**: Tighter bounds (smaller t) may require more iterations in the precise method
5. **SDP Complexity**: The optimal transport formulation scales with O(n_demands × n_grid_points²)

## Output and Analysis

The framework provides:
- **Objective Values**: Comparison of achieved objective values across methods
- **Solve Times**: Computational efficiency comparison
- **Error Analysis**: Relative errors between SDP methods and precise method
- **Visualizations**: Density function plots and efficiency comparisons
- **CSV Export**: Detailed results for further analysis

## Troubleshooting

### Common Issues

1. **Gurobi License**: Ensure you have a valid Gurobi license
2. **Memory Issues**: Reduce grid size for SDP methods if running out of memory
3. **Convergence Issues**: Increase epsilon tolerance or adjust initial parameters
4. **Integration Errors**: Try different integration tolerances or switch between scipy and torchquad
5. **SDP Solver Issues**: The optimal transport formulation can be computationally intensive; try the simple SDP method for faster results

### Performance Tips

1. Use torchquad for faster integration in the precise method
2. Start with smaller grid sizes for SDP methods and increase gradually
3. Use smaller epsilon values only when necessary for accuracy
4. Consider the simple SDP method for initial exploration
5. Consider parallel processing for multiple parameter combinations

## Research Applications

This framework is designed for research comparing:
- Computational efficiency of different approaches
- Approximation quality of discretization methods
- Scalability of methods with problem size
- Trade-offs between accuracy and speed
- Validation of SDP formulations against precise methods

## Contributing

When extending the code:
1. Maintain the modular structure
2. Add appropriate tests for new functionality
3. Update documentation for new features
4. Follow the existing coding style and conventions
5. Ensure SDP formulations are mathematically sound 