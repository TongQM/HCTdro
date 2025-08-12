import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import gurobipy as gp
import os


class CartesianGrid:
    """Cartesian grid for discretizing the square region"""
    
    def __init__(self, size: int, side_length: float = 1.0, center_x: float = 0.0, center_y: float = 0.0):
        self.size = size
        self.side_length = side_length
        self.center_x = center_x
        self.center_y = center_y
        self.x_min = center_x - side_length / 2
        self.x_max = center_x + side_length / 2
        self.y_min = center_y - side_length / 2
        self.y_max = center_y + side_length / 2
        self.x_coords = np.linspace(self.x_min, self.x_max, size + 1)
        self.y_coords = np.linspace(self.y_min, self.y_max, size + 1)
        self.cell_centers_x = (self.x_coords[:-1] + self.x_coords[1:]) / 2
        self.cell_centers_y = (self.y_coords[:-1] + self.y_coords[1:]) / 2
        self.cell_area = (side_length / size) ** 2
        self.num_cells = size ** 2
        
    def get_cell_center(self, i: int, j: int):
        return self.cell_centers_x[i], self.cell_centers_y[j]
    
    def get_cell_index(self, i: int, j: int):
        return i * self.size + j
    
    def get_grid_indices(self, cell_index: int):
        i = cell_index // self.size
        j = cell_index % self.size
        return i, j


class EmpiricalDistribution:
    """Empirical distribution on the Cartesian grid"""
    
    def __init__(self, grid: CartesianGrid):
        self.grid = grid
        self.density = np.zeros(grid.num_cells)
        self.samples = []
        
    def generate_random_samples(self, num_samples: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        x_samples = np.random.uniform(self.grid.x_min, self.grid.x_max, num_samples)
        y_samples = np.random.uniform(self.grid.y_min, self.grid.y_max, num_samples)
        self.samples = list(zip(x_samples, y_samples))
        
        for x, y in self.samples:
            i = int((x - self.grid.x_min) / (self.grid.side_length / self.grid.size))
            j = int((y - self.grid.y_min) / (self.grid.side_length / self.grid.size))
            i = min(max(i, 0), self.grid.size - 1)
            j = min(max(j, 0), self.grid.size - 1)
            cell_index = self.grid.get_cell_index(i, j)
            self.density[cell_index] += 1
            
    def normalize(self):
        total = self.density.sum()
        if total > 0:
            self.density = self.density / total
            
    def set_density(self, density):
        if len(density) == self.grid.num_cells:
            self.density = np.array(density)
        else:
            raise ValueError(f"Density must have length {self.grid.num_cells}")


class Coordinate:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return np.array([self.x, self.y])

class SquareRegion:
    def __init__(self, side_length: float, center_x: float = 0.0, center_y: float = 0.0):
        self.side_length = side_length
        self.center_x = center_x
        self.center_y = center_y
        self.x_min = center_x - side_length / 2
        self.x_max = center_x + side_length / 2
        self.y_min = center_y - side_length / 2
        self.y_max = center_y + side_length / 2

    def __repr__(self) -> str:
        return f'Square: side_length={self.side_length}, center=({self.center_x}, {self.center_y})'

class Demand:
    def __init__(self, location: Coordinate, dmd: float):
        self.location = location
        self.dmd = dmd

    def get_coordinates(self):
        return self.location.get_coordinates()

class DemandsGenerator:
    def __init__(self, region: SquareRegion, num_demands_pts: int, seed=11):
        self.region = region
        self.num_demands_pts = num_demands_pts
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        x_coords = np.random.uniform(low=self.region.x_min, high=self.region.x_max, size=self.num_demands_pts)
        y_coords = np.random.uniform(low=self.region.y_min, high=self.region.y_max, size=self.num_demands_pts)
        return np.array([Demand(Coordinate(x, y), 1) for x, y in zip(x_coords, y_coords)])

class Polyhedron:
    def __init__(self, A, b, B, c, dimension):
        self.A, self.b = A.copy(), b.copy()
        self.B, self.c = B.copy(), c.copy()
        self.dim = dimension
        self.update_constraints()

    def update_constraints(self):
        self.eq_constraints = {'type': 'eq', 'fun': lambda x: self.B @ x - self.c , 'jac': lambda _: self.B}
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x, 'jac': lambda _: -self.A}

    def add_ineq_constraint(self, ai, bi):
        # Check for near-duplicate or parallel constraints
        if self.A.shape[0] > 0:
            for aj in self.A:
                cos_sim = np.dot(ai, aj) / (np.linalg.norm(ai) * np.linalg.norm(aj) + 1e-12)
                if abs(cos_sim) > 0.9999:
                    print("[Polyhedron] Warning: New constraint nearly parallel or duplicate to an existing one. Skipping.")
                    return
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.update_constraints()

    def find_analytic_center_with_phase1(self, x0=None):
        """
        Finds the analytic center using a robust scipy optimizer.
        First, it uses a Phase I method (Gurobi) to find a strictly feasible point.
        Then, it minimizes -sum(log(s_i)) where s_i are slacks.
        """
        import sys
        import gurobipy as gp
        from scipy.optimize import minimize

        # --- Phase I: Find a strictly feasible starting point ---
        print("  [Analytic Center] Running Phase I to find a strictly feasible point...")
        n = self.dim
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("phase_I", env=env) as model:
                    x = model.addMVar(shape=n, lb=-np.inf, name="x")
                    t = model.addMVar(shape=1, lb=-np.inf, name="t")
                    model.addConstr(self.A @ x - self.b <= -t)
                    model.addConstr(self.B @ x == self.c)
                    model.setObjective(t, gp.GRB.MAXIMIZE)
                    model.optimize()

                    if model.status != gp.GRB.OPTIMAL or t.X[0] <= 1e-6:
                        print("  [Analytic Center] Warning: Could not find a strictly interior point.", flush=True)
                        if model.status == gp.GRB.OPTIMAL:
                            x0_feasible = x.X
                        else:
                            print("  [Analytic Center] FATAL: Gurobi could not find a feasible point.", flush=True)
                            return np.zeros(self.dim), -np.inf
                    else:
                        x0_feasible = x.X
                        print(f"  [Analytic Center] Phase I successful. Found point with min slack {t.X[0]:.4e}", flush=True)
        except gp.GurobiError as e:
            print(f"  [Analytic Center] Gurobi error in Phase I: {e}", flush=True)
            return np.zeros(self.dim), -np.inf

        # --- Phase II: Find the analytic center using scipy ---
        print("  [Analytic Center] Running Phase II to find analytic center...", flush=True)
        
        def objective(x):
            slacks = self.b - self.A @ x
            return -np.sum(np.log(slacks + 1e-9))

        def objective_jac(x):
            slacks = self.b - self.A @ x
            return (self.A.T / (slacks + 1e-9)).sum(axis=1)

        res = minimize(
            fun=objective,
            x0=x0_feasible,
            method='trust-constr',
            jac=objective_jac,
            constraints=[self.eq_constraints, self.ineq_constraints],
            options={'disp': False, 'maxiter': 1000}
        )

        if res.success:
            print("  [Analytic Center] Scipy successfully found the analytic center.", flush=True)
            return res.x, -res.fun
        else:
            print(f"  [Analytic Center] Warning: Scipy failed: {res.message}", flush=True)
            print("  [Analytic Center] Returning last feasible point from Phase I.", flush=True)
            return x0_feasible, -objective(x0_feasible) 

    def find_analytic_center_newton(self, x0=None, max_iters: int = 200, tol: float = 1e-8, verbose: bool = True):
        """
        Equality-constrained damped Newton method for the analytic center:
        minimize -sum_i log(b_i - a_i^T x) subject to B x = c and b - A x > 0.

        Returns (x, phi(x)) where phi is the barrier objective value.
        """
        import gurobipy as gp
        import numpy as np

        # Phase I to get strictly feasible x satisfying Bx=c and b - Ax > 0
        if verbose:
            print("  [Analytic Center] (Newton) Phase I for strictly feasible point...")
        n = self.dim
        try:
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model("phase_I_newton", env=env) as model:
                    x = model.addMVar(shape=n, lb=-np.inf, name="x")
                    t = model.addMVar(shape=1, lb=-np.inf, name="t")
                    model.addConstr(self.A @ x - self.b <= -t)
                    model.addConstr(self.B @ x == self.c)
                    model.setObjective(t, gp.GRB.MAXIMIZE)
                    model.optimize()
                    if model.status != gp.GRB.OPTIMAL or t.X[0] <= 1e-9:
                        if verbose:
                            print("  [Analytic Center] (Newton) Warning: could not certify strict interior; using boundary-feasible point")
                        xk = x.X
                    else:
                        xk = x.X
                        if verbose:
                            print(f"  [Analytic Center] (Newton) Phase I slack {t.X[0]:.3e}")
        except gp.GurobiError as e:
            if verbose:
                print(f"  [Analytic Center] (Newton) Gurobi error in Phase I: {e}")
            # Fallback: zero vector projected to equality constraints
            # Compute least-squares solution of B x = c
            BT = self.B.T
            xk = BT @ np.linalg.pinv(self.B @ BT) @ self.c

        def phi(x):
            s = self.b - self.A @ x
            if np.any(s <= 0):
                return np.inf
            return -np.sum(np.log(s))

        def grad(x, s=None):
            if s is None:
                s = self.b - self.A @ x
            return self.A.T @ (1.0 / s)

        def hessian(x, s=None):
            if s is None:
                s = self.b - self.A @ x
            W = (1.0 / (s**2))
            return self.A.T @ (W[:, None] * self.A)

        # Newton iterations
        for k in range(max_iters):
            s = self.b - self.A @ xk
            if np.any(s <= 0):
                # Step outside; shrink back in
                if verbose:
                    print("  [Analytic Center] (Newton) Infeasible iterate encountered; backing off")
                # small projection towards interior
                xk = xk * 0.9
                continue
            g = grad(xk, s)
            H = hessian(xk, s)

            # Solve KKT system for equality-constrained Newton step
            # [H  B^T; B 0] [dx; nu] = [-g; 0]
            B = self.B
            zeros = np.zeros((B.shape[0], B.shape[0]))
            top = np.concatenate([H, B.T], axis=1)
            bottom = np.concatenate([B, zeros], axis=1)
            KKT = np.concatenate([top, bottom], axis=0)
            rhs = -np.concatenate([g, np.zeros(B.shape[0])])

            # Regularize if ill-conditioned
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                if verbose:
                    print("  [Analytic Center] (Newton) KKT solve failed; adding ridge")
                ridge = 1e-8
                Hreg = H + ridge * np.eye(H.shape[0])
                top = np.concatenate([Hreg, B.T], axis=1)
                KKT = np.concatenate([top, bottom], axis=0)
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            dx = sol[:n]

            # Check convergence: small Newton decrement or step
            if np.linalg.norm(dx) < tol:
                if verbose:
                    print(f"  [Analytic Center] (Newton) Converged in {k} iterations")
                return xk, -phi(xk)

            # Fraction-to-the-boundary step size to keep s > 0
            Adx = self.A @ dx
            mask = Adx > 0
            if np.any(mask):
                alpha_max = 0.99 * np.min(s[mask] / Adx[mask])
            else:
                alpha_max = 1.0
            alpha = min(1.0, float(alpha_max))

            # Backtracking line search (Armijo)
            c1 = 1e-4
            tau = 0.5
            phi_x = phi(xk)
            gd = g @ dx
            while True:
                x_trial = xk + alpha * dx
                s_trial = self.b - self.A @ x_trial
                if np.all(s_trial > 0) and phi(x_trial) <= phi_x + c1 * alpha * gd:
                    break
                alpha *= tau
                if alpha < 1e-12:
                    # If step becomes too small, accept and stop
                    if verbose:
                        print("  [Analytic Center] (Newton) Step too small; stopping")
                    return xk, -phi_x

            xk = x_trial

        if verbose:
            print("  [Analytic Center] (Newton) Reached max iterations without convergence")
        return xk, -phi(xk)