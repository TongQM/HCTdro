import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from classes_cartesian_fixed import SquareRegion, DemandsGenerator
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp


def benchmark_accpm_times(
    region: SquareRegion,
    num_demands_list: List[int],
    t: float,
    epsilon: float,
    max_iterations: int,
    seed: int = 42,
) -> List[float]:
    """Measure ACCPM runtime vs number of demands."""
    times: List[float] = []
    for num_demands in num_demands_list:
        generator = DemandsGenerator(region, num_demands, seed=seed)
        demands = generator.generate()
        start = time.time()
        _ = find_worst_tsp_density_precise_fixed(
            region,
            demands,
            t=t,
            epsilon=epsilon,
            tol=1e-4,
            use_torchquad=True,
            max_iterations=max_iterations,
        )
        elapsed = time.time() - start
        print(f"ACCPM: n={num_demands}, time={elapsed:.2f}s")
        times.append(elapsed)
    return times


def benchmark_sdp_times(
    region: SquareRegion,
    base_num_demands: int,
    grid_sizes: List[int],
    t: float,
    seed: int = 42,
) -> List[float]:
    """Measure SDP runtime vs grid resolution (grid size)."""
    generator = DemandsGenerator(region, base_num_demands, seed=seed)
    demands = generator.generate()

    times: List[float] = []
    for grid_size in grid_sizes:
        start = time.time()
        _, info = find_worst_tsp_density_sdp(region, demands, t=t, grid_size=grid_size)
        # Prefer model-reported solve time if available
        elapsed = info["solve_time"] if info and "solve_time" in info else (time.time() - start)
        print(f"SDP: grid={grid_size}, time={elapsed:.2f}s")
        times.append(elapsed)
    return times


def plot_and_save(x, y, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    # Common setup
    region = SquareRegion(side_length=1.0)
    t = 0.3
    epsilon = 0.05
    max_iterations = 20
    seed = 42

    # ACCPM scaling with number of demands
    num_demands_list = [5, 10, 15, 20, 30]
    accpm_times = benchmark_accpm_times(
        region=region,
        num_demands_list=num_demands_list,
        t=t,
        epsilon=epsilon,
        max_iterations=max_iterations,
        seed=seed,
    )
    plot_and_save(
        x=num_demands_list,
        y=accpm_times,
        xlabel="Number of demands (n)",
        ylabel="Time (s)",
        title="ACCPM runtime vs number of demands",
        out_path="figs2/accpm_time_vs_demands.png",
    )

    # SDP scaling with grid resolution
    base_num_demands = 20
    grid_sizes = [20, 30, 40, 50, 70]
    sdp_times = benchmark_sdp_times(
        region=region,
        base_num_demands=base_num_demands,
        grid_sizes=grid_sizes,
        t=t,
        seed=seed,
    )
    plot_and_save(
        x=grid_sizes,
        y=sdp_times,
        xlabel="Grid size (resolution per axis)",
        ylabel="Time (s)",
        title=f"SDP runtime vs grid resolution (n={base_num_demands})",
        out_path="figs2/sdp_time_vs_grid.png",
    )


if __name__ == "__main__":
    main()


