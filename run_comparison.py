#!/usr/bin/env python3
"""
Script to run comparison between precise and SDP methods
and display results in a clear format.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from classes_cartesian import CartesianGrid, EmpiricalDistribution, SquareRegion, DemandsGenerator, Demand
from comparison_framework import ComparisonFramework

def main():
    print("=" * 60)
    print("COMPARISON OF PRECISE vs SDP METHODS")
    print("=" * 60)
    
    # Configuration
    config = {
        'grid_size': 6,  # Smaller grid for faster demonstration
        'epsilon': 0.1,
        'num_samples': 50,
        'random_seed': 42
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seed
    np.random.seed(config['random_seed'])
    
    # Create test problem
    print(f"\nCreating test problem...")
    grid = CartesianGrid(config['grid_size'])
    region = SquareRegion(side_length=1.0)
    # Generate random demands as Demand objects
    generator = DemandsGenerator(region, config['num_samples'], seed=config['random_seed'])
    demands = generator.generate()
    
    print(f"Grid size: {grid.size}x{grid.size} = {grid.size**2} cells")
    print(f"Number of demands: {len(demands)}")
    
    # Initialize comparison framework
    print(f"\nInitializing comparison framework...")
    framework = ComparisonFramework(region, demands, t=1, epsilon=config['epsilon'])
    
    # Run all methods with error handling
    print(f"\nRunning precise method...")
    try:
        framework.run_precise_method(use_torchquad=True, tol=1e-4)
        print("✓ Precise method completed successfully")
    except Exception as e:
        print(f"✗ Precise method failed: {str(e)}")
        framework.results['precise'] = {
            'method': 'Precise (Analytic Center Cutting Plane)',
            'error': str(e),
            'success': False
        }
    
    print(f"\nRunning SDP method...")
    try:
        framework.run_sdp_method()
        print("✓ SDP method completed successfully")
    except Exception as e:
        print(f"✗ SDP method failed: {str(e)}")
        framework.results['sdp'] = {
            'method': 'SDP',
            'error': str(e),
            'success': False
        }
    
    print(f"\nRunning improved SDP method...")
    try:
        framework.run_sdp_method_improved()
        print("✓ Improved SDP method completed successfully")
    except Exception as e:
        print(f"✗ Improved SDP method failed: {str(e)}")
        framework.results['sdp_improved'] = {
            'method': 'Improved SDP',
            'error': str(e),
            'success': False
        }
    
    print(f"\nRunning simple SDP method...")
    try:
        framework.run_sdp_method_simple()
        print("✓ Simple SDP method completed successfully")
    except Exception as e:
        print(f"✗ Simple SDP method failed: {str(e)}")
        framework.results['sdp_simple'] = {
            'method': 'Simple SDP',
            'error': str(e),
            'success': False
        }
    
    # Collect results
    results = framework.results
    
    # Display results
    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results_data = []
    for key, result in results.items():
        status = "✓" if 'objective_value' in result else "✗"
        obj_val = f"{result['objective_value']:.6f}" if 'objective_value' in result else "N/A"
        time_val = f"{result['solve_time']:.3f}s" if 'solve_time' in result else "N/A"
        error_msg = result.get('error', '') if 'error' in result else ''
        method_name = result.get('method', key)
        print(f"\n{status} {method_name}:")
        if 'objective_value' in result:
            print(f"  Objective Value: {obj_val}")
            print(f"  Computation Time: {time_val}")
        else:
            print(f"  Error: {error_msg}")
        results_data.append({
            'Method': method_name,
            'Success': 'objective_value' in result,
            'Objective_Value': result['objective_value'] if 'objective_value' in result else None,
            'Computation_Time': result['solve_time'] if 'solve_time' in result else None,
            'Error': error_msg
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_data)
    successful_results = results_df[results_df['Success'] == True]
    
    if len(successful_results) > 0:
        print(f"\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Best and worst objective values
        best_obj = successful_results['Objective_Value'].min()
        worst_obj = successful_results['Objective_Value'].max()
        best_method = successful_results.loc[successful_results['Objective_Value'].idxmin(), 'Method']
        worst_method = successful_results.loc[successful_results['Objective_Value'].idxmax(), 'Method']
        
        print(f"\nObjective Values:")
        print(f"  Best: {best_obj:.6f} ({best_method})")
        print(f"  Worst: {worst_obj:.6f} ({worst_method})")
        print(f"  Range: {worst_obj - best_obj:.6f}")
        print(f"  Relative difference: {((worst_obj - best_obj) / best_obj * 100):.2f}%")
        
        # Fastest and slowest methods
        fastest_time = successful_results['Computation_Time'].min()
        slowest_time = successful_results['Computation_Time'].max()
        fastest_method = successful_results.loc[successful_results['Computation_Time'].idxmin(), 'Method']
        slowest_method = successful_results.loc[successful_results['Computation_Time'].idxmax(), 'Method']
        
        print(f"\nComputation Times:")
        print(f"  Fastest: {fastest_time:.3f}s ({fastest_method})")
        print(f"  Slowest: {slowest_time:.3f}s ({slowest_method})")
        print(f"  Speedup factor: {slowest_time / fastest_time:.2f}x")
        
        # Efficiency analysis
        successful_results['Efficiency'] = successful_results['Objective_Value'] / successful_results['Computation_Time']
        most_efficient = successful_results.loc[successful_results['Efficiency'].idxmin(), 'Method']
        
        print(f"\nEfficiency (Objective/Time):")
        for _, row in successful_results.iterrows():
            print(f"  {row['Method']}: {row['Efficiency']:.6f}")
        print(f"  Most efficient: {most_efficient}")
        
        # Recommendations
        print(f"\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print(f"  For highest accuracy: {best_method}")
        print(f"  For fastest computation: {fastest_method}")
        print(f"  For best efficiency: {most_efficient}")
        
        if best_method == fastest_method:
            print(f"  {best_method} offers both best accuracy and speed!")
        else:
            print(f"  Trade-off between accuracy ({best_method}) and speed ({fastest_method})")
    
    else:
        print(f"\nNo methods succeeded. Check error messages above.")
    
    # Create visualizations
    if len(successful_results) > 0:
        print(f"\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Method Comparison Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Empirical distribution (not available, so show demand locations)
        ax1 = axes[0, 0]
        demand_coords = np.array([d.get_coordinates() for d in demands])
        ax1.scatter(demand_coords[:,0], demand_coords[:,1], c='blue', alpha=0.6, label='Demands')
        ax1.set_title('Empirical Demand Locations')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.legend()
        
        # Plot 2: Best worst-case density (if available)
        ax2 = axes[0, 1]
        best_result = results[[k for k in results if best_method.lower() in k.lower() or best_method in results[k].get('method','')][0]]
        density_func = best_result.get('density_function', None)
        if density_func is not None:
            # Evaluate on grid
            X, Y = np.meshgrid(np.linspace(-0.5, 0.5, grid.size), np.linspace(-0.5, 0.5, grid.size))
            Z = np.vectorize(density_func)(X, Y)
            im2 = ax2.imshow(Z, extent=[-0.5,0.5,-0.5,0.5], origin='lower', cmap='plasma')
            ax2.set_title(f'Best Worst-Case Density\n({best_method})')
            ax2.set_xlabel('X coordinate')
            ax2.set_ylabel('Y coordinate')
            plt.colorbar(im2, ax=ax2, label='Density')
        else:
            ax2.text(0.5, 0.5, 'No density\navailable', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Worst-Case Density')
        
        # Plot 3: Objective values comparison
        ax3 = axes[1, 0]
        bars = ax3.bar(successful_results['Method'], successful_results['Objective_Value'], 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax3.set_title('Objective Values (Lower is Better)')
        ax3.set_ylabel('Objective Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, successful_results['Objective_Value']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 4: Computation times comparison
        ax4 = axes[1, 1]
        bars = ax4.bar(successful_results['Method'], successful_results['Computation_Time'], 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax4.set_title('Computation Times')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, successful_results['Computation_Time']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'comparison_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {plot_filename}")
        
        # Show the plot
        plt.show()
        
        # Create comparison table
        print(f"\n" + "=" * 60)
        print("DETAILED COMPARISON TABLE")
        print("=" * 60)
        comparison_table = successful_results[['Method', 'Objective_Value', 'Computation_Time', 'Efficiency']].copy()
        comparison_table['Objective_Value'] = comparison_table['Objective_Value'].round(6)
        comparison_table['Computation_Time'] = comparison_table['Computation_Time'].round(3)
        comparison_table['Efficiency'] = comparison_table['Efficiency'].round(6)
        print(comparison_table.to_string(index=False))
    
    print(f"\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 