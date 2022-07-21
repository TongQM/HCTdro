from scipy import integrate, optimize
import numpy as np



class Polyhedron:
    def __init__(self, A, b, B, c, dimension):
        '''
        Polyhedron determined by Ax<=b form and Bx=c
        '''
        self.A, self.b = A, b
        self.B, self.c = B, c
        self.dim = dimension
        self.eq_constraints = {'type': 'eq', 'fun': lambda x: self.B @ x - self.c , 'jac': lambda _: self.B}
        # optimize.LinearConstraint(B, c, c)
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x + 1e-6, 'jac': lambda _: -self.A}
        # optimize.LinearConstraint(A, -np.inf, b + 1e-4, keep_feasible=False)

    def add_ineq_constraint(self, ai, bi):
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.ineq_constraints = optimize.LinearConstraint(self.A, -np.inf, self.b)

    def find_analytic_center(self, x0):
        objective = lambda x: -np.sum(np.log(self.b - self.A @ x + 1e-6))  # To ensure log(b - A @ x) is defined.
        objective_jac = lambda x: np.sum((self.A.T / (self.b - self.A @ x + 1e-6)), axis=1)
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=[self.ineq_constraints, self.eq_constraints], jac=objective_jac, options={'disp': True})
        assert result.success, result.message
        analytic_center, analytic_center_val = result.x, result.fun            
        return analytic_center, analytic_center_val

    def show_constraints(self):
        print(f'A: {self.A} \n b: {self.b} \n B: {self.B} \n c: {self.c}.')


class Polytope:
    def __init__(self, A, b, dimension):
        '''
        Polyhedron determined by Ax<=b form
        '''
        self.A, self.b = A, b
        self.dim = dimension
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x + 1e-6, 'jac': lambda _: -self.A}

    def add_ineq_constraint(self, ai, bi):
        self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        self.b = np.append(self.b, bi)
        self.ineq_constraints = optimize.LinearConstraint(self.A, -np.inf, self.b)

    def find_analytic_center(self, x0):
        objective = lambda x: -np.sum(np.log(self.b - self.A @ x + 1e-6))  # Add 1e-6 to ensure log(b - A @ x) is defined.
        objective_jac = lambda x: np.sum((self.A.T / (self.b - self.A @ x + 1e-6)), axis=1)
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=[self.ineq_constraints], jac=objective_jac, options={'disp': True})
        assert result.success, result.message
        analytic_center, analytic_center_val = result.x, result.fun
        analytic_center = np.append(analytic_center, -np.sum(analytic_center))
        return analytic_center, analytic_center_val

    def show_constraints(self):
        print(f'A: {self.A} \n b: {self.b}.')


def callback(xk):
    print(xk)


n = 10
diam = 10
polyhedron = Polyhedron(np.eye(n), diam*np.ones(n), np.ones((1, n)), 0, n)
polytope = Polytope(np.append(np.eye(n-1), -np.ones(n-1).reshape(1, n-1), axis=0), diam*np.ones(n), n-1)


result_equal, result_val_equal = [], []
for count in range(1000):
    print(f'Iteration {count}: \n')
    print(f'Find the analytic center of polyhedron: \n')
    polyhedron_analytic_center, polyhedron_analytic_center_val = polyhedron.find_analytic_center(np.zeros(n))
    print(f'Find the analytic center of polytope: \n')
    polytope_analytic_center, polytope_analytic_center_val = polytope.find_analytic_center(np.zeros(n-1))
    print(f'Add new constraints for polyhedron and polytope:')
    g = np.random.uniform(-2, 2, n)
    h = np.random.rand(1)
    polyhedron.add_ineq_constraint(g, h)
    modified_g = np.array([g[k] - g[-1] for k in range(n-1)])
    polytope.add_ineq_constraint(modified_g, h)
    print(f'{polyhedron_analytic_center} and {polytope_analytic_center}\n')
    print(f'{polyhedron_analytic_center_val - polytope_analytic_center_val < 0.001}\n')
    result_equal.append(np.linalg.norm(polyhedron_analytic_center - polytope_analytic_center))
    result_val_equal.append(polyhedron_analytic_center_val - polytope_analytic_center_val < 0.001)
    print(f'End of iteration {count}.\n')

