#!/usr/bin/env python3
"""
Generate feasible region heatmap comparison between ACCPM and SDP.
X-axis: Number of demands
Y-axis: Resolution (number of blocks)
Color: Success rate (fraction of successful trials)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from classes_cartesian_fixed import SquareRegion, DemandsGenerator
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp

def run_feasible_region_heatmap():
    """Generate feasible region heatmap with success rate visualization."""
    
    # Configuration
    TIME_LIMIT = 180  # 3 minutes in seconds
    N_TRIALS = 10     # Number of trials per configuration
    t = 0.5           # Fixed Wasserstein radius
    
    # Problem size ranges
    demand_counts = [5, 10, 15, 20, 25, 30, 35, 40, 50]  # Number of demands (x-axis)
    resolutions = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]  # Grid resolutions (y-axis)
    
    region = SquareRegion(side_length=10.0)
    
    print("🔍 Feasible Region Heatmap: ACCPM vs SDP (3-minute time limit)")
    print("=" * 80)
    print(f"Time limit: {TIME_LIMIT}s ({TIME_LIMIT/60:.1f} minutes)")
    print(f"Trials per configuration: {N_TRIALS}")
    print(f"Wasserstein radius: {t}")
    print(f"Service region: {region.side_length}×{region.side_length} square")
    print(f"Demand counts: {demand_counts}")
    print(f"Resolutions: {resolutions}")
    
    # Results storage - success_rate[method][demands][resolution] = success_count/N_TRIALS
    accpm_results = {}  # {n_demands: success_rate}
    sdp_results = {}    # {n_demands: {resolution: success_rate}}
    
    # Test ACCPM with different demand counts (resolution-independent)
    print(f"\n🔵 Testing ACCPM with {len(demand_counts)} demand configurations...")
    
    for n_idx, n_demands in enumerate(demand_counts):
        print(f"\n--- ACCPM {n_idx+1}/{len(demand_counts)}: {n_demands} demands ---")
        
        success_count = 0
        total_time = 0
        
        for trial in range(N_TRIALS):
            print(f"  Trial {trial+1}/{N_TRIALS}...", end=" ")
            
            try:
                # Generate random demands for this trial
                generator = DemandsGenerator(region, n_demands, seed=42+trial)
                demands = generator.generate()
                
                start_time = time.time()
                result_accpm = find_worst_tsp_density_precise_fixed(
                    region, demands, t=t, epsilon=0.01, max_iterations=50, 
                    tol=1e-3, return_params=True, return_history=True
                )
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Check success criteria
                success = False
                if elapsed_time <= TIME_LIMIT:
                    if isinstance(result_accpm, tuple) and len(result_accpm) >= 5:
                        density_func, _, _, _, history = result_accpm
                        if 'UB' in history and 'LB' in history and len(history['UB']) > 0:
                            # Find best iteration (smallest gap)
                            gaps = np.array(history['UB']) - np.array(history['LB'])
                            best_gap = np.min(gaps)
                            success = best_gap <= 0.01  # Match the epsilon parameter
                
                if success:
                    success_count += 1
                    print(f"✓ {elapsed_time:.1f}s")
                else:
                    if elapsed_time > TIME_LIMIT:
                        print(f"✗ TIMEOUT ({elapsed_time:.1f}s)")
                    else:
                        print(f"✗ NO_CONV ({elapsed_time:.1f}s)")
                        
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"✗ ERROR ({elapsed_time:.1f}s): {str(e)[:50]}")
        
        success_rate = success_count / N_TRIALS
        accpm_results[n_demands] = success_rate
        avg_time = total_time / N_TRIALS
        print(f"  Summary: {success_count}/{N_TRIALS} successful ({success_rate*100:.1f}%), avg time: {avg_time:.1f}s")
    
    # Test SDP with different demand counts and resolutions
    print(f"\n🟠 Testing SDP with {len(demand_counts)}×{len(resolutions)} configurations...")
    
    for n_idx, n_demands in enumerate(demand_counts):
        print(f"\n=== SDP Demands {n_idx+1}/{len(demand_counts)}: {n_demands} demands ===")
        sdp_results[n_demands] = {}
        
        # Generate fixed demands for this demand count
        generator = DemandsGenerator(region, n_demands, seed=42)
        demands_sdp = generator.generate()
        
        for res_idx, grid_res in enumerate(resolutions):
            print(f"\n--- Resolution {res_idx+1}/{len(resolutions)}: {grid_res}×{grid_res} ---")
            
            success_count = 0
            total_time = 0
            
            for trial in range(N_TRIALS):
                print(f"  Trial {trial+1}/{N_TRIALS}...", end=" ")
                
                try:
                    start_time = time.time()
                    sdp_result = find_worst_tsp_density_sdp(
                        region, demands_sdp, t=t, grid_size=grid_res
                    )
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    
                    # Check success criteria
                    success = False
                    if elapsed_time <= TIME_LIMIT:
                        if isinstance(sdp_result, tuple) and len(sdp_result) == 2:
                            sdp_density, info_dict = sdp_result
                            success = (sdp_density is not None and 
                                     info_dict is not None and 
                                     info_dict.get('objective_value', 0) > 0)
                    
                    if success:
                        success_count += 1
                        print(f"✓ {elapsed_time:.1f}s")
                    else:
                        if elapsed_time > TIME_LIMIT:
                            print(f"✗ TIMEOUT ({elapsed_time:.1f}s)")
                        else:
                            print(f"✗ FAILED ({elapsed_time:.1f}s)")
                            
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    print(f"✗ ERROR ({elapsed_time:.1f}s): {str(e)[:50]}")
            
            success_rate = success_count / N_TRIALS
            sdp_results[n_demands][grid_res] = success_rate
            avg_time = total_time / N_TRIALS
            print(f"  Summary: {success_count}/{N_TRIALS} successful ({success_rate*100:.1f}%), avg time: {avg_time:.1f}s")
    
    # Generate heatmap visualization
    print(f"\n📊 Generating feasible region heatmaps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data matrices
    n_demands = len(demand_counts)
    n_resolutions = len(resolutions)
    
    # ACCPM heatmap (same success rate across all resolutions for each demand count)
    accpm_matrix = np.zeros((n_resolutions, n_demands))
    for j, demands in enumerate(demand_counts):
        if demands in accpm_results:
            success_rate = accpm_results[demands]
            accpm_matrix[:, j] = success_rate  # Fill entire column with same rate
        else:
            accpm_matrix[:, j] = 0  # No data
    
    # SDP heatmap 
    sdp_matrix = np.zeros((n_resolutions, n_demands))
    for j, demands in enumerate(demand_counts):
        for i, res in enumerate(resolutions):
            if demands in sdp_results and res in sdp_results[demands]:
                sdp_matrix[i, j] = sdp_results[demands][res]
            else:
                sdp_matrix[i, j] = 0  # No data or failed
    
    # Plot ACCPM heatmap
    im1 = ax1.imshow(accpm_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
    ax1.set_title(f'Feasible region within {TIME_LIMIT/60:.0f}-min (ACCPM, template)', fontsize=14)
    ax1.set_xlabel('# demands', fontsize=12)
    ax1.set_ylabel('resolution (# of blocks)', fontsize=12)
    
    # Set ticks and labels for ACCPM
    ax1.set_xticks(range(n_demands))
    ax1.set_xticklabels(demand_counts)
    ax1.set_yticks(range(0, n_resolutions, 2))  # Show every 2nd resolution
    ax1.set_yticklabels([resolutions[i] for i in range(0, n_resolutions, 2)])
    
    # Add text annotations for ACCPM (show success rates)
    for j in range(n_demands):
        for i in range(0, n_resolutions, 3):  # Show every 3rd to avoid clutter
            demands = demand_counts[j]
            if demands in accpm_results:
                rate = accpm_results[demands]
                if rate > 0:  # Only show if there's some success
                    text_color = 'white' if rate < 0.5 else 'black'
                    ax1.text(j, i, f'{rate:.1f}', ha='center', va='center', 
                            color=text_color, fontsize=8, fontweight='bold')
    
    # Plot SDP heatmap
    im2 = ax2.imshow(sdp_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
    ax2.set_title(f'Feasible region within {TIME_LIMIT/60:.0f}-min (Direct solve, template)', fontsize=14)
    ax2.set_xlabel('# demands', fontsize=12)
    ax2.set_ylabel('resolution (# of blocks)', fontsize=12)
    
    # Set ticks and labels for SDP
    ax2.set_xticks(range(n_demands))
    ax2.set_xticklabels(demand_counts)
    ax2.set_yticks(range(0, n_resolutions, 2))
    ax2.set_yticklabels([resolutions[i] for i in range(0, n_resolutions, 2)])
    
    # Add text annotations for SDP (show success rates where significant)
    for j in range(n_demands):
        for i in range(n_resolutions):
            demands = demand_counts[j]
            res = resolutions[i]
            if demands in sdp_results and res in sdp_results[demands]:
                rate = sdp_results[demands][res]
                if rate > 0:  # Only show if there's some success
                    text_color = 'white' if rate < 0.5 else 'black'
                    # Only show text for every few cells to avoid clutter
                    if i % 2 == 0 and j % 2 == 0:
                        ax2.text(j, i, f'{rate:.1f}', ha='center', va='center', 
                                color=text_color, fontsize=8, fontweight='bold')
    
    # Add some space between subplots and move colorbar to the right
    plt.subplots_adjust(right=0.85, wspace=0.3)
    
    # Add shared colorbar on the right side
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Success rate (fraction of trials)', fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add annotations explaining the difference
    ax1.text(0.02, 0.98, 'Same success rate\nacross all resolutions', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.98, 'Different success rate\nfor each resolution', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save heatmap
    output_path = 'figs/feasible_region_heatmap.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Heatmap saved: {output_path}")
    
    # Print detailed summary
    print("\n" + "="*80)
    print("FEASIBLE REGION HEATMAP RESULTS")
    print("="*80)
    
    print(f"\n🔵 ACCPM Results (same across all resolutions):")
    print("  Demands | Success Rate | Status")
    print("  --------|--------------|--------")
    for demands in demand_counts:
        if demands in accpm_results:
            rate = accpm_results[demands]
            status = "solved" if rate >= 0.8 else "timeout/imprecise" if rate >= 0.2 else "timeout/imprecise"
            print(f"  {demands:7d} | {rate*100:10.1f}% | {status}")
        else:
            print(f"  {demands:7d} | {'N/A':>10} | not tested")
    
    print(f"\n🟠 SDP Results (varies by resolution):")
    print("  Sample results for key configurations:")
    for demands in [demand_counts[0], demand_counts[len(demand_counts)//2], demand_counts[-1]]:
        if demands in sdp_results:
            print(f"  {demands} demands:")
            for res in [resolutions[0], resolutions[len(resolutions)//2], resolutions[-1]]:
                if res in sdp_results[demands]:
                    rate = sdp_results[demands][res]
                    status = "solved" if rate >= 0.8 else "timeout/imprecise" if rate >= 0.2 else "timeout/imprecise"
                    print(f"    {res}×{res}: {rate*100:5.1f}% {status}")
    
    return accpm_results, sdp_results

if __name__ == "__main__":
    accpm_results, sdp_results = run_feasible_region_heatmap()
