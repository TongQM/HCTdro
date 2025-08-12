import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import argparse

from classes_cartesian_fixed import SquareRegion, Demand, Coordinate
from precise_method_cartesian_fixed import find_worst_tsp_density_precise_fixed
from sdp_method_cartesian import find_worst_tsp_density_sdp


def sample_truncated_mixture_gaussians(
    region: SquareRegion,
    num_points: int,
    means: List[Tuple[float, float]],
    sigmas: List[float],
    weights: List[float],
    seed: int = 42,
) -> np.ndarray:
    """
    Draw samples from a truncated Gaussian mixture restricted to the square region.
    Returns an array of shape (num_points, 2).
    """
    assert len(means) == len(sigmas) == len(weights)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)

    samples: List[np.ndarray] = []
    while len(samples) < num_points:
        # choose component
        k = rng.choice(len(weights), p=weights)
        mean = np.array(means[k], dtype=float)
        sigma = float(sigmas[k])
        # sample until inside region
        for _ in range(10000):
            point = rng.normal(loc=mean, scale=sigma, size=2)
            if (
                region.x_min <= point[0] <= region.x_max
                and region.y_min <= point[1] <= region.y_max
            ):
                samples.append(point)
                break
        # fallback (should be rare): clip to region
        if len(samples) < num_points and _ == 9999:
            point = np.clip(point, [region.x_min, region.y_min], [region.x_max, region.y_max])
            samples.append(point)

    return np.vstack(samples)[:num_points]


def build_demands(points: np.ndarray) -> np.ndarray:
    return np.array([Demand(Coordinate(float(x), float(y)), 1.0) for x, y in points])


def compare_densities_on_grid(
    region: SquareRegion,
    f1,
    f2,
    resolution: int = 200,
):
    x = np.linspace(region.x_min, region.x_max, resolution)
    y = np.linspace(region.y_min, region.y_max, resolution)
    X, Y = np.meshgrid(x, y)

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z1[j, i] = f1(X[j, i], Y[j, i])
            Z2[j, i] = f2(X[j, i], Y[j, i])

    diff = Z1 - Z2
    area = ((region.x_max - region.x_min) / (resolution - 1)) * (
        (region.y_max - region.y_min) / (resolution - 1)
    )
    l2_sq = np.sum(diff**2) * area
    linf = np.max(np.abs(diff))
    # correlation on flattened arrays
    z1f = Z1.ravel()
    z2f = Z2.ravel()
    corr = np.corrcoef(z1f, z2f)[0, 1]
    return X, Y, Z1, Z2, diff, l2_sq, linf, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=float, default=1.0, help='Wasserstein radius t')
    parser.add_argument('--grid_size', type=int, default=50, help='SDP grid size per axis')
    parser.add_argument('--num_demands', type=int, default=30, help='Number of sampled demands')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for sampling')
    args = parser.parse_args()

    # 10x10 miles square region centered at 0
    region = SquareRegion(side_length=10.0)

    # Mixture of Gaussians parameters (within the 10x10 square)
    means = [(-3.0, -2.0), (2.5, 3.0)]
    sigmas = [1.0, 1.3]
    weights = [0.6, 0.4]

    # Common parameters
    num_demands = args.num_demands
    t = args.t
    epsilon = 0.05
    grid_size = args.grid_size  # for SDP
    seed = args.seed

    # Sample nominal demands from truncated mixture
    pts = sample_truncated_mixture_gaussians(
        region=region,
        num_points=num_demands,
        means=means,
        sigmas=sigmas,
        weights=weights,
        seed=seed,
    )
    demands = build_demands(pts)

    # Run ACCPM
    print("Running ACCPM (precise) ...")
    t0 = time.time()
    f_precise = find_worst_tsp_density_precise_fixed(
        region,
        demands,
        t=t,
        epsilon=epsilon,
        tol=1e-4,
        use_torchquad=True,
        max_iterations=150,
    )
    accpm_time = time.time() - t0
    print(f"ACCPM time: {accpm_time:.2f}s")

    # Run SDP (discrete approximation)
    print("Running SDP (discrete approximation) ...")
    t1 = time.time()
    f_sdp, sdp_info = find_worst_tsp_density_sdp(region, demands, t=t, grid_size=grid_size)
    sdp_time = (sdp_info["solve_time"] if sdp_info and "solve_time" in sdp_info else time.time() - t1)
    print(f"SDP time: {sdp_time:.2f}s")

    # Compare densities on a grid
    X, Y, Zp, Zs, Zdiff, l2_sq, linf, corr = compare_densities_on_grid(region, f_precise, f_sdp, resolution=150)
    print(f"Similarity metrics: L2^2={l2_sq:.4e}, L_inf={linf:.4e}, corr={corr:.4f}")

    # Save plots
    demands_xy = pts

    plt.figure(figsize=(15, 4.5))
    # Precise
    plt.subplot(1, 3, 1)
    im1 = plt.contourf(X, Y, Zp, levels=20, cmap="viridis")
    plt.scatter(demands_xy[:, 0], demands_xy[:, 1], c="red", s=20, marker="x")
    plt.title(f"Precise (ACCPM)\nTime: {accpm_time:.2f}s")
    plt.colorbar(im1)
    # SDP
    plt.subplot(1, 3, 2)
    im2 = plt.contourf(X, Y, Zs, levels=20, cmap="viridis")
    plt.scatter(demands_xy[:, 0], demands_xy[:, 1], c="red", s=20, marker="x")
    plt.title(f"SDP (grid={grid_size})\nTime: {sdp_time:.2f}s")
    plt.colorbar(im2)
    # Difference
    plt.subplot(1, 3, 3)
    im3 = plt.contourf(X, Y, Zdiff, levels=20, cmap="coolwarm")
    plt.title(f"Difference (Precise - SDP)\nL2^2={l2_sq:.2e}, Linf={linf:.2e}, r={corr:.2f}")
    plt.colorbar(im3)
    plt.tight_layout()
    plt.savefig("figs2/mixture_precise_vs_sdp.png", dpi=150)
    print("Saved: figs2/mixture_precise_vs_sdp.png")

    # Print a simple coincidence statement
    coincide = (linf < 5e-2) or (corr > 0.98)
    print(f"Do the worst-case distributions coincide (by tolerance)? {coincide}")


if __name__ == "__main__":
    main()


