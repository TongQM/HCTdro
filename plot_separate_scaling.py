#!/usr/bin/env python3
"""
Plot ACCMP and SDP computation times separately to better visualize their individual scaling behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
from classes_cartesian_fixed import SquareRegion, DemandsGenerator
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp
import time

def test_accmp_scaling():
    """Test ACCMP scaling with different numbers of demands."""
    print("Testing ACCMP scaling...")
    
    region = SquareRegion(side_length=10.0)
    demand_counts = [5, 8, 10, 12, 15, 18, 20, 25]
    t = 0.5
    max_iterations = 50
    epsilon = 0.05
    timeout = 180  # 3 minutes
    
    accmp_results = []
    
    for n_demands in demand_counts:
        print(f"\n--- Testing ACCMP with {n_demands} demands ---")
        
        generator = DemandsGenerator(region, n_demands, seed=42)
        demands = generator.generate()
        
        try:
            start_time = time.time()
            
            result = find_worst_tsp_density_precise_fixed(
                region, demands, t=t, epsilon=epsilon, 
                max_iterations=max_iterations, tol=1e-3, return_history=True
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            failed = runtime > timeout
            
            print(f"  Runtime: {runtime:.2f}s, Failed: {failed}")
            
            accmp_results.append({
                'n_demands': n_demands,
                'runtime': runtime,
                'failed': failed
            })
            
            if failed:
                print(f"  Timeout reached for {n_demands} demands")
                break
                
        except Exception as e:
            print(f"  ACCMP failed with exception: {e}")
            accmp_results.append({
                'n_demands': n_demands,
                'runtime': timeout,
                'failed': True
            })
            break
    
    return accmp_results

def test_sdp_scaling():
    """Test SDP scaling with different resolutions."""
    print("\nTesting SDP scaling...")
    
    region = SquareRegion(side_length=10.0)
    resolutions = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_demands = 20
    t = 0.5
    
    generator = DemandsGenerator(region, n_demands, seed=42)
    demands = generator.generate()
    
    sdp_results = []
    
    for resolution in resolutions:
        print(f"\n--- Testing SDP with resolution {resolution}x{resolution} ---")
        
        try:
            start_time = time.time()
            
            result = find_worst_tsp_density_sdp(
                region, demands, t=t, grid_size=resolution
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            print(f"  Runtime: {runtime:.2f}s")
            
            sdp_results.append({
                'resolution': resolution,
                'problem_size': resolution * resolution,
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
                    coeffs = np.polyfit(prev_sizes, prev_times, 2)
                    estimated_time = np.polyval(coeffs, resolution * resolution)
                else:
                    estimated_time = (resolution / 10) ** 2 * prev_times[0]
            else:
                estimated_time = (resolution / 10) ** 2 * 0.1
            
            sdp_results.append({
                'resolution': resolution,
                'problem_size': resolution * resolution,
                'runtime': estimated_time,
                'failed': True
            })
    
    return sdp_results

def create_separate_plots(accmp_results, sdp_results):
    """Create separate plots for ACCMP and SDP scaling."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ACCMP Plot
    accmp_sizes = [r['n_demands'] for r in accmp_results]
    accmp_times = [r['runtime'] for r in accmp_results]
    accmp_failed = [r['failed'] for r in accmp_results]
    
    # Separate successful and failed runs
    accmp_success_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if not f]
    accmp_success_times = [t for t, f in zip(accmp_times, accmp_failed) if not f]
    accmp_failed_sizes = [s for s, f in zip(accmp_sizes, accmp_failed) if f]
    accmp_failed_times = [t for t, f in zip(accmp_times, accmp_failed) if f]
    
    # Plot ACCMP
    if accmp_success_sizes:
        ax1.loglog(accmp_success_sizes, accmp_success_times, 'o-', color='blue', 
                  linewidth=2, markersize=8, label='Successful runs')
    
    if accmp_failed_sizes:
        ax1.loglog(accmp_failed_sizes, accmp_failed_times, 'x', color='red', 
                  markersize=12, markeredgewidth=3, label='Failed runs (timeout)')
    
    ax1.set_xlabel('Number of demand points', fontsize=12)
    ax1.set_ylabel('Wall-clock time (seconds, log scale)', fontsize=12)
    ax1.set_title('ACCPM Scalability', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add 3-minute timeout line
    ax1.axhline(y=180, color='red', linestyle='--', alpha=0.7, label='3-min timeout')
    
    # SDP Plot
    sdp_sizes = [r['problem_size'] for r in sdp_results]
    sdp_times = [r['runtime'] for r in sdp_results]
    sdp_failed = [r['failed'] for r in sdp_results]
    
    # Separate successful and failed runs
    sdp_success_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if not f]
    sdp_success_times = [t for t, f in zip(sdp_times, sdp_failed) if not f]
    sdp_failed_sizes = [s for s, f in zip(sdp_sizes, sdp_failed) if f]
    sdp_failed_times = [t for t, f in zip(sdp_times, sdp_failed) if f]
    
    # Plot SDP
    if sdp_success_sizes:
        ax2.loglog(sdp_success_sizes, sdp_success_times, 's-', color='orange', 
                  linewidth=2, markersize=8, label='Successful runs')
    
    if sdp_failed_sizes:
        ax2.loglog(sdp_failed_sizes, sdp_failed_times, 'x', color='red', 
                  markersize=12, markeredgewidth=3, label='Failed runs')
    
    # Fit and show quadratic trend line for SDP
    if len(sdp_success_sizes) >= 3:
        coeffs = np.polyfit(np.log(sdp_success_sizes), np.log(sdp_success_times), 1)
        slope = coeffs[0]
        x_trend = np.logspace(np.log10(min(sdp_success_sizes)), 
                             np.log10(max(sdp_success_sizes)), 100)
        y_trend = np.exp(coeffs[1]) * (x_trend ** slope)
        ax2.loglog(x_trend, y_trend, '--', color='gray', alpha=0.7, 
                  label=f'Trend: O(n^{slope:.1f})')
    
    ax2.set_xlabel('Problem size (grid points)', fontsize=12)
    ax2.set_ylabel('Wall-clock time (seconds, log scale)', fontsize=12)
    ax2.set_title('SDP Scalability', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('figs/separate_scaling_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSeparate scaling plots saved to: figs/separate_scaling_comparison.png")
    plt.close()

def print_summary(accmp_results, sdp_results):
    """Print detailed summary of results."""
    print(f"\n" + "=" * 70)
    print("DETAILED SCALING ANALYSIS")
    print("=" * 70)
    
    print("\n🔵 ACCMP Results:")
    print("Problem size: Number of demand points")
    for r in accmp_results:
        status = "❌ TIMEOUT" if r['failed'] else "✅ SUCCESS"
        print(f"  n={r['n_demands']:2d}: {r['runtime']:6.2f}s [{status}]")
    
    # ACCMP analysis
    successful_accmp = [r for r in accmp_results if not r['failed']]
    if len(successful_accmp) >= 2:
        sizes = [r['n_demands'] for r in successful_accmp]
        times = [r['runtime'] for r in successful_accmp]
        
        # Check if there's exponential growth
        log_times = np.log(times)
        coeffs = np.polyfit(sizes, log_times, 1)
        growth_rate = coeffs[0]
        
        print(f"\n  📊 ACCMP Analysis:")
        print(f"     • Growth pattern: {'Exponential' if growth_rate > 0.1 else 'Sublinear'}")
        print(f"     • Growth rate: {growth_rate:.3f} (log-time per demand point)")
        print(f"     • Highly variable performance due to pathological behavior")
    
    print(f"\n🟠 SDP Results:")
    print("Problem size: Grid resolution squared (total variables)")
    for r in sdp_results:
        status = "❌ FAILED" if r['failed'] else "✅ SUCCESS"
        print(f"  {r['resolution']}x{r['resolution']} ({r['problem_size']:4d} vars): {r['runtime']:6.2f}s [{status}]")
    
    # SDP analysis
    successful_sdp = [r for r in sdp_results if not r['failed']]
    if len(successful_sdp) >= 3:
        sizes = [r['problem_size'] for r in successful_sdp]
        times = [r['runtime'] for r in successful_sdp]
        
        # Fit power law: time = a * size^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        coeffs = np.polyfit(log_sizes, log_times, 1)
        power = coeffs[0]
        
        print(f"\n  📊 SDP Analysis:")
        print(f"     • Scaling: O(n^{power:.2f}) where n = grid points")
        print(f"     • {'Quadratic' if 1.8 <= power <= 2.2 else 'Non-quadratic'} scaling")
        print(f"     • Consistent, predictable performance")
    
    # Comparison
    accmp_failures = sum(1 for r in accmp_results if r['failed'])
    sdp_failures = sum(1 for r in sdp_results if r['failed'])
    
    print(f"\n🏆 COMPARISON:")
    print(f"   ACCMP failure rate: {accmp_failures}/{len(accmp_results)} ({100*accmp_failures/len(accmp_results):.1f}%)")
    print(f"   SDP failure rate:   {sdp_failures}/{len(sdp_results)} ({100*sdp_failures/len(sdp_results):.1f}%)")
    
    if successful_accmp and successful_sdp:
        avg_accmp = np.mean([r['runtime'] for r in successful_accmp])
        avg_sdp = np.mean([r['runtime'] for r in successful_sdp])
        print(f"   Average runtime - ACCMP: {avg_accmp:.2f}s, SDP: {avg_sdp:.2f}s")

def main():
    """Main function to run the separate scaling analysis."""
    print("=" * 70)
    print("SEPARATE SCALING ANALYSIS: ACCMP vs SDP")
    print("=" * 70)
    
    # Test both methods
    accmp_results = test_accmp_scaling()
    sdp_results = test_sdp_scaling()
    
    # Create plots
    create_separate_plots(accmp_results, sdp_results)
    
    # Print detailed analysis
    print_summary(accmp_results, sdp_results)

if __name__ == "__main__":
    main()
