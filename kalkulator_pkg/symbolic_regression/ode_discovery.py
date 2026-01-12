"""ODE Discovery Engine for Symbolic Regression.

This module enables the genetic engine to discover differential equations
like y'' + y = 0 (which defines sin/cos) instead of curve-fitting.

The key insight: Instead of finding f(x) such that f(x) ≈ y,
we find f(y, y', y'') such that f(y, y', y'') ≈ 0.

This allows the engine to "rediscover trigonometry" from first principles.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass, field

from .numerical_diff import compute_derivatives, resample_to_even_spacing, check_even_spacing
from .expression_tree import ExpressionTree, ExpressionNode, NodeType, BINARY_OPERATORS
from .pareto_front import ParetoFront, ParetoSolution


@dataclass
class ODEConfig:
    """Configuration for ODE Discovery."""
    
    population_size: int = 200
    generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    parsimony_coefficient: float = 0.01
    
    # ODE-specific settings
    max_order: int = 2  # Maximum derivative order (2 = y'')
    early_stop_residual: float = 1e-6  # Stop if residual < this
    
    # Non-triviality constraint: minimum coefficient for highest derivative
    min_leading_coeff: float = 0.1
    
    verbose: bool = True


class ODEDiscoveryEngine:
    """Engine to discover differential equations from data.
    
    Instead of fitting y = f(x), this engine searches for relationships
    of the form f(y, y', y'') = 0.
    
    Example: Given sin(x) data, discovers y'' + y = 0.
    """
    
    def __init__(self, config: ODEConfig | None = None):
        """Initialize the ODE discovery engine.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or ODEConfig()
        self.pareto_front = ParetoFront()
        
        # Variable names for the ODE
        # y, y', y'' become y0, y1, y2 internally
        self.var_names = ['y0', 'y1', 'y2'][:self.config.max_order + 1]
    
    def _create_random_tree(self, depth: int = 3) -> ExpressionTree:
        """Create a random expression tree using y, y', y''.
        
        Uses restricted operators: +, -, *, and constants.
        No division (to avoid singularities) or complex ops.
        """
        import random
        
        operators = ['add', 'sub', 'mul']
        
        def build_node(current_depth: int) -> ExpressionNode:
            if current_depth >= depth or (current_depth > 0 and random.random() < 0.3):
                # Terminal: variable or constant
                if random.random() < 0.7:
                    var = random.choice(self.var_names)
                    return ExpressionNode(
                        node_type=NodeType.VARIABLE,
                        value=var,
                        children=[]
                    )
                else:
                    # Simple integer constants
                    const = random.choice([-2, -1, 0, 1, 2])
                    return ExpressionNode(
                        node_type=NodeType.CONSTANT,
                        value=float(const),
                        children=[]
                    )
            else:
                # Operator node
                op = random.choice(operators)
                left = build_node(current_depth + 1)
                right = build_node(current_depth + 1)
                return ExpressionNode(
                    node_type=NodeType.BINARY_OP,
                    value=op,
                    children=[left, right]
                )
        
        root = build_node(0)
        return ExpressionTree(root=root, variables=self.var_names)
    
    def _evaluate_ode(
        self, 
        tree: ExpressionTree,
        y: np.ndarray,
        y_prime: np.ndarray,
        y_double_prime: np.ndarray
    ) -> np.ndarray:
        """Evaluate an ODE expression on derivative data.
        
        The expression uses variables y0, y1, y2 for y, y', y''.
        We want f(y, y', y'') ≈ 0.
        """
        # Build data matrix
        n = len(y)
        X = np.column_stack([y, y_prime, y_double_prime])
        
        try:
            result = tree.evaluate(X)
            return result
        except Exception:
            return np.full(n, np.inf)
    
    def _calculate_fitness(
        self,
        tree: ExpressionTree,
        y: np.ndarray,
        y_prime: np.ndarray,
        y_double_prime: np.ndarray
    ) -> float:
        """Calculate fitness for an ODE candidate.
        
        Fitness = MSE from zero + parsimony penalty + triviality penalty.
        """
        try:
            # Evaluate: we want f(y, y', y'') = 0
            residual = self._evaluate_ode(tree, y, y_prime, y_double_prime)
            
            if not np.all(np.isfinite(residual)):
                return float('inf')
            
            # MSE from zero (we want the expression to equal 0)
            mse = float(np.mean(residual**2))
            
            # NON-TRIVIALITY CHECK
            # If the expression is essentially constant (evaluates same for all data),
            # it might be a trivial solution like 0*y = 0
            if np.std(residual) < 1e-10 and np.mean(np.abs(residual)) < 1e-10:
                # Check if it's the trivial zero solution
                # by evaluating at random different points
                test_y = np.array([1.0, 2.0, 3.0])
                test_y_prime = np.array([0.5, 1.5, 2.5])
                test_y_double = np.array([0.1, 0.2, 0.3])
                test_result = self._evaluate_ode(tree, test_y, test_y_prime, test_y_double)
                
                if np.all(np.abs(test_result) < 1e-10):
                    # Trivial solution (always 0)
                    return float('inf')
            
            # Parsimony penalty
            complexity = tree.complexity()
            penalty = self.config.parsimony_coefficient * complexity
            
            return mse + penalty
            
        except Exception:
            return float('inf')
    
    def _initialize_population(self, n: int) -> list[ExpressionTree]:
        """Initialize population with random ODE candidates.
        
        Also seeds with common physics patterns.
        """
        population = []
        
        # Seed with known physics patterns
        physics_seeds = [
            "y2 + y0",        # y'' + y = 0 (harmonic oscillator / trig)
            "y1 - y0",        # y' - y = 0 (exponential growth)
            "y1 + y0",        # y' + y = 0 (exponential decay)
            "y2 - y0",        # y'' - y = 0 (hyperbolic)
            "y2",             # y'' = 0 (linear)
            "y1",             # y' = 0 (constant)
        ]
        
        for seed_str in physics_seeds:
            try:
                local_dict = {v: sp.Symbol(v) for v in self.var_names}
                expr = sp.sympify(seed_str, locals=local_dict)
                tree = ExpressionTree.from_sympy(expr, self.var_names)
                population.append(tree)
            except Exception:
                pass
        
        # Fill rest with random trees
        while len(population) < n:
            tree = self._create_random_tree(depth=3)
            population.append(tree)
        
        return population[:n]
    
    def _tournament_select(
        self,
        population: list[ExpressionTree],
        fitnesses: list[float]
    ) -> ExpressionTree:
        """Select best individual from random tournament."""
        import random
        
        indices = random.sample(range(len(population)), min(self.config.tournament_size, len(population)))
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[str, float]:
        """Discover an ODE from data.
        
        Args:
            x: Independent variable values (must be evenly spaced)
            y: Dependent variable values
            
        Returns:
            Tuple of (ode_string, residual_mse)
        """
        from .operators import point_mutation, crossover
        import random
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Check/resample for even spacing
        is_even, _ = check_even_spacing(x)
        if not is_even:
            if self.config.verbose:
                print("Resampling data to even spacing...")
            x, y = resample_to_even_spacing(x, y, n_points=min(50, len(x)))
        
        # Compute derivatives
        x_int, y_int, y_prime, y_double_prime = compute_derivatives(x, y, validate_spacing=False)
        
        if self.config.verbose:
            print(f"ODE Discovery: {len(y_int)} interior points after differentiation")
        
        # Initialize population
        population = self._initialize_population(self.config.population_size)
        
        # Evolution loop
        best_tree = None
        best_fitness = float('inf')
        
        for gen in range(self.config.generations):
            # Evaluate fitness
            fitnesses = [
                self._calculate_fitness(tree, y_int, y_prime, y_double_prime)
                for tree in population
            ]
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_tree = population[gen_best_idx].copy()
            
            # Progress output
            if self.config.verbose and gen % 10 == 0:
                best_expr = best_tree.to_pretty_string() if best_tree else "?"
                print(f"Gen {gen}: Best residual {best_fitness:.2e} ({best_expr[:40]})")
            
            # Early stop
            if best_fitness < self.config.early_stop_residual:
                if self.config.verbose:
                    print(f"Early stop: residual {best_fitness:.2e}")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best
            new_population.append(best_tree.copy())
            
            while len(new_population) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    parent1 = self._tournament_select(population, fitnesses)
                    parent2 = self._tournament_select(population, fitnesses)
                    try:
                        # crossover returns tuple of (child1, child2)
                        child1, child2 = crossover(parent1, parent2)
                        child = child1  # Use first offspring
                    except Exception:
                        child = parent1.copy()
                else:
                    child = self._tournament_select(population, fitnesses)
                
                if random.random() < self.config.mutation_rate:
                    try:
                        # Use simplified operators for ODE
                        child = point_mutation(
                            child, 
                            mutation_rate=0.3,
                            operators=['add', 'sub', 'mul']
                        )
                    except Exception:
                        pass
                
                new_population.append(child)
            
            population = new_population
        
        # Format result
        if best_tree:
            ode_str = self._format_ode(best_tree)
            return ode_str, best_fitness
        else:
            return "?", float('inf')
    
    def _format_ode(self, tree: ExpressionTree) -> str:
        """Format an ODE expression tree as a human-readable string.
        
        Converts: y2 + y0 → y'' + y = 0
        """
        expr_str = tree.to_pretty_string()
        
        # Replace internal variable names with standard notation
        expr_str = expr_str.replace('y2', "y''")
        expr_str = expr_str.replace('y1', "y'")
        expr_str = expr_str.replace('y0', 'y')
        
        return f"{expr_str} = 0"

    def discover_autonomous_ode(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[str, float]:
        """Discover autonomous ODE y' = G(y) using phase space regression.
        
        Instead of fitting f(y, y', y'') = 0, this method:
        1. Computes (y, y') pairs from data
        2. Runs regression with y as input and y' as target
        3. Returns the discovered G(y) function
        
        This detects ODEs like y' = y(1-y) for logistic sigmoid.
        
        Args:
            x: Independent variable values
            y: Dependent variable values
            
        Returns:
            Tuple of (ode_string "y' = G(y)", residual_mse)
        """
        from .genetic_engine import GeneticSymbolicRegressor, GeneticConfig
        
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Resample if needed
        is_even, _ = check_even_spacing(x)
        if not is_even:
            x, y = resample_to_even_spacing(x, y, n_points=min(50, len(x)))
        
        # Compute y' (first derivative only)
        x_int, y_int, y_prime, _ = compute_derivatives(x, y, validate_spacing=False)
        
        if len(y_int) < 5:
            return "?", float('inf')
        
        # Phase space regression: y is input, y' is target
        # This finds G(y) such that y' = G(y)
        Y_input = y_int.reshape(-1, 1)  # y as feature
        Y_target = y_prime  # y' as target
        
        # Create config for quick regression
        config = GeneticConfig(
            population_size=100,
            generations=30,
            verbose=False,
            parsimony_coefficient=0.02,
            # Allow polynomial terms for y(1-y) = y - y²
            operators=['add', 'sub', 'mul', 'square', 'neg']
        )
        
        regressor = GeneticSymbolicRegressor(config)
        pareto = regressor.fit(Y_input, Y_target, ['y'])
        
        best = pareto.get_best()
        if best:
            ode_str = f"y' = {best.expression}"
            return ode_str, best.mse
        else:
            return "?", float('inf')

