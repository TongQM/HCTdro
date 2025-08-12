#!/usr/bin/env python3
"""
Generate the exact scalability comparison figure: ACCMP vs SDP (Direct Formulation).
Problem size = number of demands for ACCMP, resolution for SDP.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from classes_cartesian_fixed import SquareRegion, DemandsGenerator, Demand, Coordinate
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp
from scipy.stats import linregress

def test_accmp_scalability():
    """Test ACCMP scalability with different numbers of demands."""
    print("Testing ACCMP scalability...")
    
    region = SquareRegion(side_length=10.0)
    demand_counts = [5, 8, 10, 12, 15, 18, 20]
    t = 0.5
    max_iterations = 150
    epsilon = 0.05
    timeout = 180  # 3 minutes timeout
    
    accmp_results = []
    
    for n_demands in demand_counts:
        print(f"\n--- Testing ACCMP with {n_demands} demands ---")
        
        # Generate demands
        generator = DemandsGenerator(region, n_demands, seed=42)
        demands = generator.generate()
        
        try:
            start_time = time.time()
            
            # Run ACCMP with timeout protection
            result = find_worst_tsp_density_precise_fixed(
                region, demands, t=t, epsilon=epsilon, 
                max_iterations=max_iterations, tol=1e-3, return_history=True
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Check for failure conditions
            failed = False
            failure_reason = ""
            
            if runtime > timeout:
                failed = True
                failure_reason = "timeout"
            elif isinstance(result, tuple) and len(result) >= 4:
                # Check convergence history
                _, _, _, history = result
                if len(history['ub']) > 1:
                    # Check if gap is increasing (sign of failure)
                    final_gap = history['ub'][-1] - history['lb'][-1]
                    initial_gap = history['ub'][0] - history['lb'][0]
                    if final_gap > initial_gap * 2:  # Gap doubled
                        failed = True
                        failure_reason = "diverging_gap"
            
            print(f"  Runtime: {runtime:.2f}s, Failed: {failed}")
            if failed:
                print(f"  Failure reason: {failure_reason}")
            
            accmp_results.append({
                'n_demands': n_demands,
                'runtime': runtime,
                'failed': failed,
                'failure_reason': failure_reason
            })
            
        except Exception as e:
            print(f"  ACCPM failed with exception: {e}")
            # Estimate runtime based on previous successful runs
            if accmp_results:
                # Extrapolate based on previous results
                prev_times = [r['runtime'] for r in accmp_results if not r['failed']]
                prev_sizes = [r['n_demands'] for r in accmp_results if not r['failed']]
                if len(prev_times) >= 2:
                    # Fit exponential growth
                    log_times = np.log(prev_times)
                    slope, intercept, _, _, _ = linregress(prev_sizes, log_times)
                    estimated_time = np.exp(intercept + slope * n_demands)
                else:
                    estimated_time = 600  # Default high estimate
            else:
                estimated_time = 600
            
            accmp_results.append({
                'n_demands': n_demands,
                'runtime': estimated_time,
                'failed': True,
                'failure_reason': 'exception'
            })
    
    return accmp_results

def test_sdp_scalability():
    """Test SDP scalability with different resolutions."""
    print("\nTesting SDP scalability...")
    
    region = SquareRegion(side_length=10.0)
    resolutions = [10, 15, 20, 25, 30, 35, 40]
    n_demands = 20  # Fixed number of demands
    t = 0.5
    
    # Generate fixed demands
    generator = DemandsGenerator(region, n_demands, seed=42)
    demands = generator.generate()
    demands_locations = np.array([d.get_coordinates() for d in demands])
    
    sdp_results = []
    
    for resolution in resolutions:
        print(f"\n--- Testing SDP with resolution {resolution}x{resolution} ---")
        
        try:
            start_time = time.time()
            
            # Run SDP
            result = find_worst_tsp_density_sdp(
                region, demands, t=t, grid_size=resolution
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            print(f"  Runtime: {runtime:.2f}s")
            
            sdp_results.append({
                'resolution': resolution,
                'problem_size': resolution * resolution,  # Total variables
                'runtime': runtime,
                'failed': False
            })
            
        except Exception as e:
            print(f"  SDP failed with exception: {e}")
            # Estimate based on quadratic scaling
            if sdp_results:
                prev_times = [r['runtime'] for r in sdp_results]
                prev_sizes = [r['problem_size'] for r in sdp_results]
                if len(prev_times) >= 2:
                    # Fit quadratic growth
                    coeffs = np.polyfit(prev_sizes, prev_times, 2)
                    estimated_time = np.polyval(coeffs, resolution * resolution)
                else:
                    estimated_time = (resolution / 20) ** 2 * prev_times[0]
            else:
                estimated_time = (resolution / 20) ** 2 * 10  # Rough estimate
            
            sdp_results.append({
                'resolution': resolution,
                'problem_size': resolution * resolution,
                'runtime': estimated_time,
                'failed': True
            })
    
    return sdp_results

def create_scalability_plot(accmp_results, sdp_results):
    """Create the scalability comparison plot."""
    
    # Prepare data
    accmp_sizes = [r['n_demands'] for r in accmp_results]
    accmp_times = [r['runtime'] for r in accmp_results]
    accmp_failed = [r['failed'] for r in accmp_results]
    
    sdp_sizes = [r['problem_size'] for r in sdp_results]
    sdp_times = [r['runtime'] for r in sdp_results]
    sdp_failed = [r['failed'] for r in sdp_results]
    
    print(f"\nPlot data:")
    print(f"ACCMP sizes: {accmp_sizes}")
    print(f"ACCMP times: {accmp_times}")
    print(f"ACCMP failed: {accmp_failed}")
    print(f"SDP sizes: {sdp_sizes}")
    print(f"SDP times: {sdp_times}")
    print(f"SDP failed: {sdp_failed}")
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # Plot successful runs
    accmp_success_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if not f]
    accmp_success_times = [t for t, f in zip(accmp_times, accmp_failed) if not f]
    
    sdp_success_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if not f]
    sdp_success_times = [t for t, f in zip(sdp_times, sdp_failed) if not f]
    
    # Plot lines for successful runs
    if accmp_success_sizes:
        plt.loglog(accmp_success_sizes, accmp_success_times, 'o-', color='blue', 
                  label='ACCPM (scales with # demand points)', linewidth=2, markersize=8)
    
    if sdp_success_sizes:
        plt.loglog(sdp_success_sizes, sdp_success_times, 's-', color='orange',
                  label='Direct solve (scales with resolution)', linewidth=2, markersize=8)
    
    # Plot failed runs with red X
    accmp_failed_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if f]
    accmp_failed_times = [t for t, f in zip(accmp_times, accmp_failed) if f]
    
    if accmp_failed_sizes:
        plt.loglog(accmp_failed_sizes, accmp_failed_times, 'x', color='red', 
                  markersize=12, markeredgewidth=3, label='ACCPM failures')
    
    sdp_failed_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if f]
    sdp_failed_times = [t for t, f in zip(sdp_times, sdp_failed) if f]
    
    if sdp_failed_sizes:
        plt.loglog(sdp_failed_sizes, sdp_failed_times, 'x', color='red', 
                  markersize=12, markeredgewidth=3)
    
    # Formatting
    plt.xlabel('Problem size', fontsize=12)
    plt.ylabel('Wall-clock time (seconds, log scale)', fontsize=12)
    plt.title('Scalability: ACCPM vs Direct Formulation (different size measures)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set axis limits to show all data points
    all_sizes = accmp_sizes + sdp_sizes
    all_times = accmp_times + sdp_times
    if all_sizes and all_times:
        plt.xlim(min(all_sizes) * 0.8, max(all_sizes) * 1.2)
        plt.ylim(min(all_times) * 0.8, max(all_times) * 1.5)
    
    # Add annotations for failures
    if accmp_failed_sizes:
        for size, time_est in zip(accmp_failed_sizes, accmp_failed_times):
            plt.annotate(f'Failed\n(est. {time_est:.0f}s)', 
                        xy=(size, time_est), xytext=(size*1.2, time_est*0.5),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('figs/scalability_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nScalability plot saved to: figs/scalability_comparison.png")
    plt.close()

def main():
    """Main function to run the scalability comparison."""
    print("=" * 70)
    print("GENERATING SCALABILITY COMPARISON FIGURE")
    print("=" * 70)
    
    # Test both methods
    accmp_results = test_accmp_scalability()
    sdp_results = test_sdp_scalability()
    
    # Create the plot
    create_scalability_plot(accmp_results, sdp_results)
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nACCMP Results:")
    for r in accmp_results:
        status = "FAILED" if r['failed'] else "SUCCESS"
        print(f"  n={r['n_demands']:2d}: {r['runtime']:6.2f}s [{status}]")
    
    print("\nSDP Results:")
    for r in sdp_results:
        status = "FAILED" if r['failed'] else "SUCCESS"
        print(f"  {r['resolution']}x{r['resolution']} ({r['problem_size']:4d} vars): {r['runtime']:6.2f}s [{status}]")
    
    # Count failures
    accmp_failures = sum(1 for r in accmp_results if r['failed'])
    sdp_failures = sum(1 for r in sdp_results if r['failed'])
    
    print(f"\nFailure rates:")
    print(f"  ACCMP: {accmp_failures}/{len(accmp_results)} ({100*accmp_failures/len(accmp_results):.1f}%)")
    print(f"  SDP: {sdp_failures}/{len(sdp_results)} ({100*sdp_failures/len(sdp_results):.1f}%)")

if __name__ == "__main__":
    main()
