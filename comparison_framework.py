import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import integrate
from classes_cartesian import SquareRegion, Coordinate, DemandsGenerator, Demand
from precise_method_cartesian import find_worst_tsp_density_precise
from sdp_method_cartesian import find_worst_tsp_density_sdp, find_worst_tsp_density_sdp_improved, find_worst_tsp_density_sdp_simple
import seaborn as sns


class ComparisonFramework:
    def __init__(self, region: SquareRegion, demands, t: float = 1, epsilon: float = 0.1):
        self.region = region
        self.demands = demands
        self.t = t
        self.epsilon = epsilon
        self.results = {}
        
    def run_precise_method(self, use_torchquad: bool = True, tol: float = 1e-4):
        """Run the precise method (analytic center cutting plane)"""
        print("=" * 50)
        print("RUNNING PRECISE METHOD")
        print("=" * 50)
        
        start_time = time.time()
        density_function = find_worst_tsp_density_precise(
            self.region, self.demands, self.t, self.epsilon, tol, use_torchquad
        )
        end_time = time.time()
        
        # Compute objective value
        objective_value = self.compute_objective_value(density_function)
        
        self.results['precise'] = {
            'method': 'Precise (Analytic Center Cutting Plane)',
            'density_function': density_function,
            'objective_value': objective_value,
            'solve_time': end_time - start_time,
            'use_torchquad': use_torchquad,
            'tol': tol
        }
        
        print(f"Precise method completed in {end_time - start_time:.2f} seconds")
        print(f"Objective value: {objective_value:.6f}")
        
        return density_function, objective_value, end_time - start_time
    
    def run_sdp_method(self, grid_size: int = 50):
        """Run the SDP method (corrected formulation)"""
        print("=" * 50)
        print("RUNNING SDP METHOD")
        print("=" * 50)
        
        density_function, sdp_info = find_worst_tsp_density_sdp(
            self.region, self.demands, self.t, grid_size
        )
        
        if density_function is not None:
            # Compute objective value
            objective_value = self.compute_objective_value(density_function)
            
            self.results['sdp'] = {
                'method': 'SDP (Optimal Transport)',
                'density_function': density_function,
                'objective_value': objective_value,
                'solve_time': sdp_info['solve_time'],
                'grid_size': grid_size,
                'sdp_info': sdp_info
            }
            
            print(f"SDP method completed in {sdp_info['solve_time']:.2f} seconds")
            print(f"Objective value: {objective_value:.6f}")
            
            return density_function, objective_value, sdp_info['solve_time']
        else:
            print("SDP method failed")
            return None, None, None
    
    def run_sdp_improved_method(self, grid_size: int = 50):
        """Run the improved SDP method"""
        print("=" * 50)
        print("RUNNING IMPROVED SDP METHOD")
        print("=" * 50)
        
        density_function, sdp_info = find_worst_tsp_density_sdp_improved(
            self.region, self.demands, self.t, grid_size
        )
        
        if density_function is not None:
            # Compute objective value
            objective_value = self.compute_objective_value(density_function)
            
            self.results['sdp_improved'] = {
                'method': 'SDP Improved (Optimal Transport)',
                'density_function': density_function,
                'objective_value': objective_value,
                'solve_time': sdp_info['solve_time'],
                'grid_size': grid_size,
                'sdp_info': sdp_info
            }
            
            print(f"Improved SDP method completed in {sdp_info['solve_time']:.2f} seconds")
            print(f"Objective value: {objective_value:.6f}")
            
            return density_function, objective_value, sdp_info['solve_time']
        else:
            print("Improved SDP method failed")
            return None, None, None
    
    def run_sdp_simple_method(self, grid_size: int = 50):
        """Run the simple SDP method"""
        print("=" * 50)
        print("RUNNING SIMPLE SDP METHOD")
        print("=" * 50)
        
        density_function, sdp_info = find_worst_tsp_density_sdp_simple(
            self.region, self.demands, self.t, grid_size
        )
        
        if density_function is not None:
            # Compute objective value
            objective_value = self.compute_objective_value(density_function)
            
            self.results['sdp_simple'] = {
                'method': 'SDP Simple (Approximate)',
                'density_function': density_function,
                'objective_value': objective_value,
                'solve_time': sdp_info['solve_time'],
                'grid_size': grid_size,
                'sdp_info': sdp_info
            }
            
            print(f"Simple SDP method completed in {sdp_info['solve_time']:.2f} seconds")
            print(f"Objective value: {objective_value:.6f}")
            
            return density_function, objective_value, sdp_info['solve_time']
        else:
            print("Simple SDP method failed")
            return None, None, None
    
    def compute_objective_value(self, density_function):
        """Compute the objective value: int_R sqrt(f(x)) dA"""
        def integrand(x, y):
            return np.sqrt(density_function(x, y))
        
        result, _ = integrate.dblquad(
            lambda y, x: integrand(x, y),
            self.region.x_min, self.region.x_max,
            lambda _: self.region.y_min, lambda _: self.region.y_max,
            epsabs=1e-4
        )
        return result
    
    def compare_methods(self, methods_to_run=['precise', 'sdp', 'sdp_improved', 'sdp_simple'], 
                       grid_sizes=[30, 50, 100], use_torchquad=True):
        """Run all methods and compare results"""
        print("=" * 60)
        print("COMPARISON FRAMEWORK")
        print("=" * 60)
        print(f"Region: {self.region}")
        print(f"Number of demands: {len(self.demands)}")
        print(f"Wasserstein bound t: {self.t}")
        print(f"Epsilon: {self.epsilon}")
        print("=" * 60)
        
        comparison_results = []
        
        # Run precise method
        if 'precise' in methods_to_run:
            self.run_precise_method(use_torchquad=use_torchquad)
            if 'precise' in self.results:
                comparison_results.append({
                    'method': 'Precise',
                    'objective_value': self.results['precise']['objective_value'],
                    'solve_time': self.results['precise']['solve_time'],
                    'grid_size': 'N/A',
                    'error': 0.0
                })
        
        # Run SDP methods with different grid sizes
        for grid_size in grid_sizes:
            if 'sdp' in methods_to_run:
                self.run_sdp_method(grid_size)
                if 'sdp' in self.results:
                    precise_obj = self.results.get('precise', {}).get('objective_value', 0)
                    sdp_obj = self.results['sdp']['objective_value']
                    error = abs(sdp_obj - precise_obj) / precise_obj if precise_obj > 0 else 0
                    
                    comparison_results.append({
                        'method': f'SDP (grid={grid_size})',
                        'objective_value': sdp_obj,
                        'solve_time': self.results['sdp']['solve_time'],
                        'grid_size': grid_size,
                        'error': error
                    })
            
            if 'sdp_improved' in methods_to_run:
                self.run_sdp_improved_method(grid_size)
                if 'sdp_improved' in self.results:
                    precise_obj = self.results.get('precise', {}).get('objective_value', 0)
                    sdp_improved_obj = self.results['sdp_improved']['objective_value']
                    error = abs(sdp_improved_obj - precise_obj) / precise_obj if precise_obj > 0 else 0
                    
                    comparison_results.append({
                        'method': f'SDP Improved (grid={grid_size})',
                        'objective_value': sdp_improved_obj,
                        'solve_time': self.results['sdp_improved']['solve_time'],
                        'grid_size': grid_size,
                        'error': error
                    })
            
            if 'sdp_simple' in methods_to_run:
                self.run_sdp_simple_method(grid_size)
                if 'sdp_simple' in self.results:
                    precise_obj = self.results.get('precise', {}).get('objective_value', 0)
                    sdp_simple_obj = self.results['sdp_simple']['objective_value']
                    error = abs(sdp_simple_obj - precise_obj) / precise_obj if precise_obj > 0 else 0
                    
                    comparison_results.append({
                        'method': f'SDP Simple (grid={grid_size})',
                        'objective_value': sdp_simple_obj,
                        'solve_time': self.results['sdp_simple']['solve_time'],
                        'grid_size': grid_size,
                        'error': error
                    })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_results)
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(df.to_string(index=False))
        
        return df
    
    def plot_density_comparison(self, resolution=100):
        """Plot density functions from different methods"""
        if not self.results:
            print("No results to plot. Run comparison first.")
            return
        
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        x = np.linspace(self.region.x_min, self.region.x_max, resolution)
        y = np.linspace(self.region.y_min, self.region.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        for i, (method_name, result) in enumerate(self.results.items()):
            density_func = result['density_function']
            Z = np.zeros_like(X)
            
            for ix in range(resolution):
                for iy in range(resolution):
                    Z[iy, ix] = density_func(X[iy, ix], Y[iy, ix])
            
            im = axes[i].contourf(X, Y, Z, levels=20, cmap='viridis')
            axes[i].set_title(f"{result['method']}\nObj: {result['objective_value']:.4f}\nTime: {result['solve_time']:.2f}s")
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            
            # Plot demand points
            demands_coords = np.array([d.get_coordinates() for d in self.demands])
            axes[i].scatter(demands_coords[:, 0], demands_coords[:, 1], c='red', s=50, marker='x', label='Demands')
            axes[i].legend()
            
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def plot_efficiency_comparison(self, df):
        """Plot efficiency comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time vs Grid Size
        sdp_data = df[df['method'].str.contains('SDP')]
        if not sdp_data.empty:
            grid_sizes = sdp_data['grid_size'].values
            times = sdp_data['solve_time'].values
            methods = sdp_data['method'].values
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, method in enumerate(set(methods)):
                mask = methods == method
                ax1.plot(grid_sizes[mask], times[mask], 'o-', label=method, color=colors[i % len(colors)])
            
            ax1.set_xlabel('Grid Size')
            ax1.set_ylabel('Solve Time (seconds)')
            ax1.set_title('Solve Time vs Grid Size')
            ax1.legend()
            ax1.grid(True)
        
        # Error vs Grid Size
        if not sdp_data.empty:
            errors = sdp_data['error'].values
            
            for i, method in enumerate(set(methods)):
                mask = methods == method
                ax2.plot(grid_sizes[mask], errors[mask], 's-', label=method, color=colors[i % len(colors)])
            
            ax2.set_xlabel('Grid Size')
            ax2.set_ylabel('Relative Error')
            ax2.set_title('Error vs Grid Size')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='comparison_results.csv'):
        """Save comparison results to CSV"""
        if not self.results:
            print("No results to save. Run comparison first.")
            return
        
        # Create summary DataFrame
        summary_data = []
        for method_name, result in self.results.items():
            summary_data.append({
                'method': result['method'],
                'objective_value': result['objective_value'],
                'solve_time': result['solve_time'],
                'grid_size': result.get('grid_size', 'N/A'),
                'use_torchquad': result.get('use_torchquad', 'N/A'),
                'tol': result.get('tol', 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


def run_comprehensive_comparison(region_size=2.0, num_demands=10, t=0.5, epsilon=0.1, 
                                grid_sizes=[30, 50, 100], seed=42):
    """Run a comprehensive comparison with different parameters"""
    
    # Create region and generate demands
    region = SquareRegion(side_length=region_size)
    generator = DemandsGenerator(region, num_demands, seed=seed)
    demands = generator.generate()
    
    print(f"Generated {num_demands} demand points in {region_size}x{region_size} square region")
    
    # Create comparison framework
    framework = ComparisonFramework(region, demands, t, epsilon)
    
    # Run comparison
    df = framework.compare_methods(
        methods_to_run=['precise', 'sdp', 'sdp_improved', 'sdp_simple'],
        grid_sizes=grid_sizes,
        use_torchquad=True
    )
    
    # Plot results
    framework.plot_density_comparison()
    framework.plot_efficiency_comparison(df)
    
    # Save results
    framework.save_results()
    
    return framework, df


if __name__ == "__main__":
    # Example usage
    framework, results = run_comprehensive_comparison(
        region_size=2.0,
        num_demands=8,
        t=0.3,
        epsilon=0.1,
        grid_sizes=[20, 30, 50]
    ) 