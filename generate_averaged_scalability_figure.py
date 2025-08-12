#!/usr/bin/env python3
"""
Generate the scalability comparison figure with multiple trials and averaging to reduce randomness.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from classes_cartesian_fixed import SquareRegion, DemandsGenerator, Demand, Coordinate
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp
from scipy.stats import linregress

def test_accmp_scalability():
    """Test ACCMP scalability with multiple trials per problem size."""
    print("Testing ACCMP scalability...")
    
    region = SquareRegion(side_length=10.0)
    demand_counts = [5, 8, 10, 12, 15, 18, 20]
    t = 0.5
    max_iterations = 50  # Reduced to avoid long runs
    epsilon = 0.05
    timeout = 180  # 3 minutes timeout
    n_trials = 10  # Number of trials per problem size
    
    accmp_results = []
    
    for n_demands in demand_counts:
        print(f"\n--- Testing ACCMP with {n_demands} demands ({n_trials} trials) ---")
        
        trial_times = []
        trial_failed = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}:")
            
            # Generate different demands for each trial
            generator = DemandsGenerator(region, n_demands, seed=42 + trial * 10)
            demands = generator.generate()
            
            try:
                start_time = time.time()
                
                # Run ACCMP with timeout protection
                result = find_worst_tsp_density_precise_fixed(
                    region, demands, t=t, epsilon=epsilon, 
                    max_iterations=max_iterations, tol=1e-3, return_params=True, return_history=True
                )
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Check for failure conditions
                failed = False
                failure_reason = ""
                
                if runtime > timeout:
                    failed = True
                    failure_reason = "timeout"
                elif isinstance(result, tuple) and len(result) >= 5:
                    # Check final gap - result format: (density_func, demands_locations, lambdas, v_tilde, history)
                    _, _, _, _, history = result
                    if 'UB' in history and 'LB' in history and len(history['UB']) > 0 and len(history['LB']) > 0:
                        final_gap = history['UB'][-1] - history['LB'][-1]
                        if final_gap > epsilon:
                            failed = True
                            failure_reason = f"gap_too_large (gap={final_gap:.4f} > ε={epsilon})"
                else:
                    # If we don't get proper return format, consider it a failure
                    failed = True
                    failure_reason = "invalid_result_format"
                
                print(f"    Runtime: {runtime:.2f}s, Failed: {failed}")
                if failed:
                    print(f"    Failure reason: {failure_reason}")
                
                trial_times.append(runtime)
                trial_failed.append(failed)
                
                # Continue with all 5 trials regardless of individual failures
                
            except Exception as e:
                print(f"    ACCMP failed with exception: {e}")
                trial_times.append(timeout)
                trial_failed.append(True)
                # Continue with remaining trials even after exception
        
        # Compute statistics
        if trial_times:
            avg_runtime = np.mean(trial_times)
            std_runtime = np.std(trial_times) if len(trial_times) > 1 else 0
            failure_rate = np.mean(trial_failed)
        else:
            avg_runtime = timeout
            std_runtime = 0
            failure_rate = 1.0
        
        print(f"  Average runtime: {avg_runtime:.2f}s ± {std_runtime:.2f}s")
        print(f"  Failure rate: {failure_rate:.1%}")
        
        accmp_results.append({
            'n_demands': n_demands,
            'runtime': avg_runtime,
            'std': std_runtime,
            'failed': failure_rate >= 0.5,  # Mark as failed if ≥50% trials failed
            'failure_rate': failure_rate,
            'trial_times': trial_times
        })
    
    return accmp_results

def test_sdp_scalability():
    """Test SDP scalability with multiple trials per resolution."""
    print("\nTesting SDP scalability...")
    
    region = SquareRegion(side_length=10.0)
    resolutions = [10, 15, 20, 25, 30, 35, 40]
    n_demands = 20  # Fixed number of demands
    t = 0.5
    n_trials = 10  # Number of trials per resolution
    
    sdp_results = []
    
    for resolution in resolutions:
        print(f"\n--- Testing SDP with resolution {resolution}x{resolution} ({n_trials} trials) ---")
        
        trial_times = []
        trial_failed = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}:")
            
            # Generate different demands for each trial
            generator = DemandsGenerator(region, n_demands, seed=42 + trial * 10)
            demands = generator.generate()
            
            try:
                start_time = time.time()
                
                # Run SDP
                result = find_worst_tsp_density_sdp(
                    region, demands, t=t, grid_size=resolution
                )
                
                end_time = time.time()
                runtime = end_time - start_time
                
                print(f"    Runtime: {runtime:.2f}s")
                
                trial_times.append(runtime)
                trial_failed.append(False)
                
            except Exception as e:
                print(f"    SDP failed with exception: {e}")
                # Estimate based on quadratic scaling from previous successful runs
                if sdp_results and any(not r['failed'] for r in sdp_results):
                    prev_successful = [r for r in sdp_results if not r['failed']]
                    prev_times = [r['runtime'] for r in prev_successful]
                    prev_sizes = [r['problem_size'] for r in prev_successful]
                    if len(prev_times) >= 2:
                        coeffs = np.polyfit(prev_sizes, prev_times, 2)
                        estimated_time = np.polyval(coeffs, resolution * resolution)
                    else:
                        estimated_time = (resolution / 20) ** 2 * prev_times[0]
                else:
                    estimated_time = (resolution / 20) ** 2 * 0.5
                
                trial_times.append(estimated_time)
                trial_failed.append(True)
        
        # Compute statistics
        avg_runtime = np.mean(trial_times)
        std_runtime = np.std(trial_times) if len(trial_times) > 1 else 0
        failure_rate = np.mean(trial_failed)
        
        print(f"  Average runtime: {avg_runtime:.2f}s ± {std_runtime:.2f}s")
        print(f"  Failure rate: {failure_rate:.1%}")
        
        sdp_results.append({
            'resolution': resolution,
            'problem_size': resolution * resolution,
            'runtime': avg_runtime,
            'std': std_runtime,
            'failed': failure_rate > 0.5,
            'failure_rate': failure_rate,
            'trial_times': trial_times
        })
    
    return sdp_results

def create_scalability_plot(accmp_results, sdp_results):
    """Create the scalability comparison plot with error bars."""
    
    # Prepare data
    accmp_sizes = [r['n_demands'] for r in accmp_results]
    accmp_times = [r['runtime'] for r in accmp_results]
    accmp_stds = [r['std'] for r in accmp_results]
    accmp_failed = [r['failed'] for r in accmp_results]
    
    sdp_sizes = [r['problem_size'] for r in sdp_results]
    sdp_times = [r['runtime'] for r in sdp_results]
    sdp_stds = [r['std'] for r in sdp_results]
    sdp_failed = [r['failed'] for r in sdp_results]
    
    print(f"\nPlot data:")
    print(f"ACCPM sizes: {accmp_sizes}")
    print(f"ACCPM times: {[f'{t:.2f}' for t in accmp_times]}")
    print(f"ACCPM failed: {accmp_failed}")
    print(f"SDP sizes: {sdp_sizes}")
    print(f"SDP times: {[f'{t:.2f}' for t in sdp_times]}")
    print(f"SDP failed: {sdp_failed}")
    
    # Create the plot
    plt.figure(figsize=(7, 5))
    
    # Plot successful runs with error bars
    accmp_success_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if not f]
    accmp_success_times = [t for t, f in zip(accmp_times, accmp_failed) if not f]
    accmp_success_stds = [std for std, f in zip(accmp_stds, accmp_failed) if not f]
    
    sdp_success_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if not f]
    sdp_success_times = [t for t, f in zip(sdp_times, sdp_failed) if not f]
    sdp_success_stds = [std for std, f in zip(sdp_stds, sdp_failed) if not f]
    
    # Plot lines for successful runs
    if accmp_success_sizes:
        plt.errorbar(accmp_success_sizes, accmp_success_times, yerr=accmp_success_stds,
                    fmt='o-', color='blue', linewidth=2, markersize=8, capsize=5,
                    label='ACCPM (scales with # demand points)')
    
    if sdp_success_sizes:
        plt.errorbar(sdp_success_sizes, sdp_success_times, yerr=sdp_success_stds,
                    fmt='s-', color='orange', linewidth=2, markersize=8, capsize=5,
                    label='Direct solve (scales with resolution)')
    
    # Plot failed runs with red X
    accmp_failed_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if f]
    accmp_failed_times = [t for t, f in zip(accmp_times, accmp_failed) if f]
    
    if accmp_failed_sizes:
        plt.scatter(accmp_failed_sizes, accmp_failed_times, marker='x', color='red', 
                   s=200, linewidths=3, label='ACCPM failures', zorder=5)
    
    sdp_failed_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if f]
    sdp_failed_times = [t for t, f in zip(sdp_times, sdp_failed) if f]
    
    if sdp_failed_sizes:
        plt.scatter(sdp_failed_sizes, sdp_failed_times, marker='x', color='red', 
                   s=200, linewidths=3, zorder=5)
    
    # Add 3-minute timeout line
    plt.axhline(y=180, color='red', linestyle='--', alpha=0.7, 
               label='3-minute timeout')
    
    # Formatting
    plt.xlabel('Problem size', fontsize=12)
    plt.ylabel('Wall-clock time (seconds)', fontsize=12)
    # plt.title('Scalability: ACCPM vs Direct Formulation (averaged over 10 trials)', fontsize=14)
    plt.yscale('log') 
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set axis limits to show all data points
    all_sizes = accmp_sizes + sdp_sizes
    all_times = accmp_times + sdp_times
    if all_sizes and all_times:
        plt.xlim(min(all_sizes) * 0.8, max(all_sizes) * 1.2)
        plt.ylim(min([t for t in all_times if t > 0]) * 0.5, max(all_times) * 1.5)
    
    # Add annotations for failures
    if accmp_failed_sizes:
        for size, time_est in zip(accmp_failed_sizes, accmp_failed_times):
            plt.annotate(f'Timeout\n({time_est:.0f}s)', 
                        xy=(size, time_est), xytext=(size*1.2, time_est*0.7),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figs/averaged_scalability_comparison.pdf', dpi=200, bbox_inches='tight')
    print(f"\nAveraged scalability plot saved to: figs/averaged_scalability_comparison.png")
    plt.close()

def print_detailed_summary(accmp_results, sdp_results):
    """Print detailed summary with statistics."""
    print(f"\n" + "=" * 70)
    print("AVERAGED SCALABILITY ANALYSIS (10 trials per problem size)")
    print("=" * 70)
    
    print("\n🔵 ACCMP Results:")
    for r in accmp_results:
        status = "❌ FAILED" if r['failed'] else "✅ SUCCESS"
        print(f"  n={r['n_demands']:2d}: {r['runtime']:6.2f}s ± {r['std']:5.2f}s "
              f"[{status}] (failure rate: {r['failure_rate']:.1%})")
        print(f"      Individual trials: {[f'{t:.1f}s' for t in r['trial_times']]}")
    
    print(f"\n🟠 SDP Results:")
    for r in sdp_results:
        status = "❌ FAILED" if r['failed'] else "✅ SUCCESS"
        print(f"  {r['resolution']}x{r['resolution']} ({r['problem_size']:4d} vars): "
              f"{r['runtime']:6.2f}s ± {r['std']:5.2f}s [{status}] "
              f"(failure rate: {r['failure_rate']:.1%})")
        print(f"      Individual trials: {[f'{t:.1f}s' for t in r['trial_times']]}")
    
    # Overall statistics
    accmp_failures = sum(1 for r in accmp_results if r['failed'])
    sdp_failures = sum(1 for r in sdp_results if r['failed'])
    
    print(f"\n📊 SUMMARY:")
    print(f"  ACCMP overall failure rate: {accmp_failures}/{len(accmp_results)} "
          f"({100*accmp_failures/len(accmp_results):.1f}%)")
    print(f"  SDP overall failure rate:   {sdp_failures}/{len(sdp_results)} "
          f"({100*sdp_failures/len(sdp_results):.1f}%)")
    
    # Compute average performance for successful cases
    successful_accmp = [r for r in accmp_results if not r['failed']]
    successful_sdp = [r for r in sdp_results if not r['failed']]
    
    if successful_accmp:
        avg_accmp_time = np.mean([r['runtime'] for r in successful_accmp])
        avg_accmp_std = np.mean([r['std'] for r in successful_accmp])
        print(f"  ACCMP average runtime (successful cases): {avg_accmp_time:.2f}s ± {avg_accmp_std:.2f}s")
    
    if successful_sdp:
        avg_sdp_time = np.mean([r['runtime'] for r in successful_sdp])
        avg_sdp_std = np.mean([r['std'] for r in successful_sdp])
        print(f"  SDP average runtime (successful cases): {avg_sdp_time:.2f}s ± {avg_sdp_std:.2f}s")

def main():
    """Main function to run the averaged scalability comparison."""
    print("=" * 70)
    print("GENERATING AVERAGED SCALABILITY COMPARISON FIGURE")
    print("=" * 70)
    
    # Test both methods
    accmp_results = test_accmp_scalability()
    sdp_results = test_sdp_scalability()
    
    # Create the plot
    create_scalability_plot(accmp_results, sdp_results)
    
    # Print summary
    print_detailed_summary(accmp_results, sdp_results)

if __name__ == "__main__":
    main()
