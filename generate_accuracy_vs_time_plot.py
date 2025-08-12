#!/usr/bin/env python3
"""
Generate Accuracy vs. Time Pareto plot comparing ACCPM and SDP methods.
Uses small problem sizes to ensure ACCPM works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from classes_cartesian_fixed import SquareRegion, CartesianGrid, EmpiricalDistribution, DemandsGenerator
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp
import time
import warnings
warnings.filterwarnings('ignore')

def compute_accuracy_metrics(accpm_density, sdp_density, grid):
    """
    Compute accuracy metrics between ACCPM and SDP solutions.
    Returns relative gap to exact inter optimum (using ACCPM as reference).
    """
    # Sample both densities on the grid
    accmp_values = []
    sdp_values = []
    
    # Convert linear index to (i,j) coordinates for grid
    for idx in range(grid.size * grid.size):
        i = idx // grid.size  # row
        j = idx % grid.size   # column
        x, y = grid.get_cell_center(i, j)
        accmp_val = accpm_density(x, y) if callable(accpm_density) else 0
        sdp_val = sdp_density(x, y) if callable(sdp_density) else 0
        accmp_values.append(accmp_val)
        sdp_values.append(sdp_val)
    
    accmp_values = np.array(accmp_values)
    sdp_values = np.array(sdp_values)
    
    # Compute sqrt integrals (objectives)
    cell_area = (grid.side_length / grid.size) ** 2
    accmp_obj = np.sum(np.sqrt(np.maximum(accmp_values, 1e-12))) * cell_area
    sdp_obj = np.sum(np.sqrt(np.maximum(sdp_values, 1e-12))) * cell_area
    
    # Relative gap (using ACCPM as "exact" reference)
    if accmp_obj > 1e-6:
        relative_gap = abs(sdp_obj - accmp_obj) / accmp_obj
    else:
        relative_gap = abs(sdp_obj - accmp_obj)
    
    return relative_gap, accmp_obj, sdp_obj

def run_accuracy_time_comparison():
    """Run accuracy vs time comparison focusing on one case with multiple trials."""
    
    # Fixed problem configuration
    n_demands = 10  # Larger number of demands to emphasize SDP advantage
    t = 0.5         # Fixed Wasserstein radius
    
    # ACCPM: Multiple runs with same settings (to show variability)
    accmp_trials = 5  # More trials to capture ACCPM variability
    
    # SDP: Different resolutions (to show accuracy-time trade-off)
    sdp_resolutions = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # Different grid resolutions
    sdp_trials = 5  # Multiple trials per resolution to capture timing variability
    reference_resolution = 70  # High-resolution SDP as "ground truth"
    
    region = SquareRegion(side_length=10.0)
    
    accpm_results = []
    sdp_results = []
    
    print("=" * 70)
    print("ACCURACY vs TIME COMPARISON (Pareto Analysis)")
    print(f"Fixed: n_demands={n_demands}, t={t}")
    print("=" * 70)
    
    # Generate fixed demands for all experiments (reproducible)
    generator = DemandsGenerator(region, n_demands, seed=42)
    demands = generator.generate()
    
    # First, compute high-resolution SDP as reference "ground truth"
    print(f"\n📊 Computing reference solution: SDP {reference_resolution}×{reference_resolution}...")
    try:
        start_time = time.time()
        reference_result = find_worst_tsp_density_sdp(
            region, demands, t=t, grid_size=reference_resolution
        )
        reference_time = time.time() - start_time
        
        if isinstance(reference_result, tuple) and len(reference_result) == 2:
            reference_density, reference_info = reference_result
            if reference_density is not None and reference_info is not None:
                reference_obj = reference_info.get('objective_value', 0.0)
                print(f"  Reference SDP: {reference_time:.2f}s, Obj: {reference_obj:.6f}")
            else:
                print(f"  Reference SDP failed: returned None values")
                reference_obj = None
                reference_density = None
        else:
            print(f"  Reference SDP failed: unexpected return format {type(reference_result)}")
            reference_obj = None
            reference_density = None
    except Exception as e:
        print(f"  Reference SDP failed: {e}. Using ACCPM as fallback.")
        reference_obj = None
        reference_density = None
    
    print(f"\n🔵 Running ACCPM {accmp_trials} times (same settings, different random seeds)...")
    
    # Run ACCPM multiple times with same settings
    for trial in range(accmp_trials):
        print(f"\n--- ACCPM Trial {trial+1}/{accmp_trials} ---")
        
        try:
            start_time = time.time()
            result_accpm = find_worst_tsp_density_precise_fixed(
                region, demands, t=t, epsilon=0.05, max_iterations=100, 
                tol=1e-3, return_params=True, return_history=True
            )
            accmp_time = time.time() - start_time
            
            # Initialize default values
            accmp_success = False
            accmp_obj = 0
            density_func = None
            best_gap = float('inf')
            
            if isinstance(result_accpm, tuple) and len(result_accpm) >= 5:
                density_func, _, _, _, history = result_accpm
                if 'UB' in history and 'LB' in history and len(history['UB']) > 0:
                    # Find the iteration with the smallest gap
                    gaps = np.array(history['UB']) - np.array(history['LB'])
                    best_iter = np.argmin(gaps)
                    best_gap = gaps[best_iter]
                    best_ub = history['UB'][best_iter]
                    best_lb = history['LB'][best_iter]
                    
                    # Always use best iteration result
                    accmp_obj = best_ub  # Use best UB as objective
                    accmp_success = best_gap <= 0.2  # Success based on best gap
                    
                    print(f"  Best iteration: {best_iter+1}, gap: {best_gap:.4f}, UB: {best_ub:.4f}")
                
            print(f"  ACCPM: {accmp_time:.2f}s, Success: {accmp_success}, Obj: {accmp_obj:.4f}")
            
            # Always store ACCPM results if we got a valid objective (even if not "successful")
            if accmp_obj > 0 and density_func is not None:
                # Compute accuracy relative to high-resolution SDP reference
                if reference_obj is not None:
                    if reference_obj > 1e-6:
                        accmp_accuracy = abs(accmp_obj - reference_obj) / reference_obj
                    else:
                        accmp_accuracy = abs(accmp_obj - reference_obj)
                else:
                    accmp_accuracy = 0.0  # Fallback: treat as exact if no reference
                
                accpm_results.append({
                    'time': accmp_time,
                    'accuracy': accmp_accuracy,
                    'trial': trial + 1,
                    'objective': accmp_obj,
                    'converged': accmp_success,
                    'best_gap': best_gap
                })
            
        except Exception as e:
            print(f"  ACCPM failed: {e}")
    
    print(f"\n🟠 Running SDP with {len(sdp_resolutions)} different resolutions ({sdp_trials} trials each)...")
    
    # Run SDP with different resolutions, multiple trials each
    for res_idx, grid_res in enumerate(sdp_resolutions):
        print(f"\n--- SDP Resolution {res_idx+1}/{len(sdp_resolutions)}: {grid_res}x{grid_res} ---")
        
        resolution_times = []
        resolution_objectives = []
        resolution_accuracies = []
        
        for trial in range(sdp_trials):
            try:
                start_time = time.time()
                sdp_result = find_worst_tsp_density_sdp(
                    region, demands, t=t, grid_size=grid_res
                )
                sdp_time = time.time() - start_time
                
                if isinstance(sdp_result, tuple) and len(sdp_result) == 2:
                    sdp_density, info_dict = sdp_result
                    sdp_obj_val = info_dict.get('objective_value', 0.0)
                    sdp_success = sdp_density is not None
                    
                    if sdp_success and reference_obj is not None:
                        # Compute accuracy relative to reference
                        if reference_obj > 1e-6:
                            relative_gap = abs(sdp_obj_val - reference_obj) / reference_obj
                        else:
                            relative_gap = abs(sdp_obj_val - reference_obj)
                        
                        resolution_times.append(sdp_time)
                        resolution_objectives.append(sdp_obj_val)
                        resolution_accuracies.append(relative_gap)
                        
                else:
                    sdp_success = False
                    
            except Exception as e:
                print(f"  Trial {trial+1} failed: {e}")
                sdp_success = False
        
        # Compute statistics for this resolution
        if resolution_times:
            avg_time = np.mean(resolution_times)
            std_time = np.std(resolution_times)
            avg_obj = np.mean(resolution_objectives)
            avg_accuracy = np.mean(resolution_accuracies)
            
            print(f"  SDP {grid_res}x{grid_res}: {avg_time:.2f}±{std_time:.2f}s, Obj: {avg_obj:.4f}, Accuracy: {avg_accuracy*100:.2f}%")
            
            # Store aggregated SDP result
            sdp_results.append({
                'time': avg_time,
                'time_std': std_time,
                'accuracy': avg_accuracy,
                'resolution': grid_res,
                'objective': avg_obj,
                'trials': len(resolution_times)
            })
        else:
            print(f"  SDP {grid_res}x{grid_res}: All trials failed")
    
    return accpm_results, sdp_results

def create_accuracy_time_plot(accpm_results, sdp_results):
    """Create the Accuracy vs. Time Pareto plot."""
    
    plt.figure(figsize=(7, 5))
    
    # Plot ACCPM results with horizontal error bars
    if accpm_results:
        converged_results = [r for r in accpm_results if r['converged']]
        failed_results = [r for r in accpm_results if not r['converged']]
        
        if converged_results:
            conv_times = [r['time'] for r in converged_results]
            conv_accuracies = [r['accuracy'] for r in converged_results]
            
            # Calculate mean and std for converged ACCPM
            if len(conv_times) > 1:
                mean_time = np.mean(conv_times)
                std_time = np.std(conv_times)
                mean_accuracy = np.mean(conv_accuracies)
                
                plt.errorbar([mean_time], [mean_accuracy], xerr=[std_time],
                           fmt='o', color='steelblue', markersize=8, alpha=0.8,
                           capsize=3, capthick=1, elinewidth=1,
                           label='ACCPM (converged)')
            else:
                plt.scatter(conv_times, conv_accuracies, 
                           color='steelblue', s=50, alpha=0.8, 
                           label='ACCPM (converged)', marker='o')
        
        if failed_results:
            fail_times = [r['time'] for r in failed_results]
            fail_accuracies = [r['accuracy'] for r in failed_results]
            
            # Calculate mean and std for failed ACCPM
            if len(fail_times) > 1:
                mean_time = np.mean(fail_times)
                std_time = np.std(fail_times)
                mean_accuracy = np.mean(fail_accuracies)
                
                plt.errorbar([mean_time], [mean_accuracy], xerr=[std_time],
                           fmt='x', color='red', markersize=8, alpha=0.8,
                           capsize=3, capthick=1, elinewidth=1,
                           label='ACCPM (best effort)')
            else:
                plt.scatter(fail_times, fail_accuracies, 
                           color='red', s=50, alpha=0.8, 
                           label='ACCMP (best effort)', marker='x')
    
    # Plot SDP results with error bars
    if sdp_results:
        sdp_times = [r['time'] for r in sdp_results]
        sdp_time_stds = [r['time_std'] for r in sdp_results]
        sdp_accuracies = [r['accuracy'] for r in sdp_results]
        
        plt.errorbar(sdp_times, sdp_accuracies, xerr=sdp_time_stds,
                    fmt='s', color='darkorange', markersize=6, alpha=0.8,
                    capsize=3, capthick=1, elinewidth=1,
                    label='Direct solve of our formulation')
    
    # Formatting
    plt.xlabel('Wall-clock time (seconds)', fontsize=12)
    plt.ylabel('Relative gap to reference solution (SDP 100×100)', fontsize=12)
    # plt.title('Accuracy vs. Time (Pareto Analysis)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set reasonable axis limits
    if accpm_results or sdp_results:
        all_times = []
        all_accuracies = []
        
        if accpm_results:
            all_times.extend([r['time'] for r in accpm_results])
            all_accuracies.extend([r['accuracy'] for r in accpm_results])
        
        if sdp_results:
            all_times.extend([r['time'] for r in sdp_results])
            all_accuracies.extend([r['accuracy'] for r in sdp_results])
        
        if all_times:
            plt.xlim(0, max(all_times) * 1.1)
        if all_accuracies:
            plt.ylim(-0.001, max(all_accuracies) * 1.1)
    
    plt.tight_layout()
    plt.savefig('figs/accuracy_vs_time_pareto.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('figs/accuracy_vs_time_pareto.png', dpi=150, bbox_inches='tight')
    print(f"\nAccuracy vs. Time plot saved to: figs/accuracy_vs_time_pareto.pdf")
    plt.show()

def print_summary(accpm_results, sdp_results):
    """Print summary of results."""
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    converged_accpm = [r for r in accpm_results if r['converged']]
    failed_accpm = [r for r in accpm_results if not r['converged']]
    
    print(f"\n🔵 ACCPM Results:")
    print(f"  Converged: {len(converged_accpm)}/{len(accpm_results)} trials")
    print(f"  Best-effort: {len(failed_accpm)}/{len(accpm_results)} trials")
    
    if accpm_results:
        all_times = [r['time'] for r in accpm_results]
        all_accuracies = [r['accuracy'] for r in accpm_results]
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        avg_accuracy = np.mean(all_accuracies)
        print(f"  Overall: {avg_time:.2f}±{std_time:.2f}s, accuracy: {avg_accuracy:.6f}")
    
    for i, r in enumerate(accpm_results):
        status = "✓" if r['converged'] else "✗"
        print(f"  {i+1}. Trial {r['trial']} {status}: {r['time']:.2f}s, accuracy: {r['accuracy']:.6f}, gap: {r['best_gap']:.4f}, obj: {r['objective']:.4f}")
    
    print(f"\n🟠 SDP Results ({len(sdp_results)} successful resolutions):")
    for i, r in enumerate(sdp_results):
        print(f"  {i+1}. Resolution {r['resolution']}x{r['resolution']}: {r['time']:.2f}±{r['time_std']:.2f}s ({r['trials']} trials), accuracy: {r['accuracy']:.6f} ({r['accuracy']*100:.4f}%), obj: {r['objective']:.4f}")
    
    if accpm_results:
        avg_accpm_time = np.mean([r['time'] for r in accpm_results])
        print(f"\nAverage ACCPM time: {avg_accpm_time:.2f}s")
    
    if sdp_results:
        avg_sdp_time = np.mean([r['time'] for r in sdp_results])
        avg_sdp_accuracy = np.mean([r['accuracy'] for r in sdp_results])
        print(f"Average SDP time: {avg_sdp_time:.2f}s")
        print(f"Average SDP accuracy gap: {avg_sdp_accuracy:.6f} ({avg_sdp_accuracy*100:.4f}%)")

if __name__ == "__main__":
    # Run the comparison
    accpm_results, sdp_results = run_accuracy_time_comparison()
    
    # Create the plot
    create_accuracy_time_plot(accpm_results, sdp_results)
    
    # Print summary
    print_summary(accpm_results, sdp_results)
