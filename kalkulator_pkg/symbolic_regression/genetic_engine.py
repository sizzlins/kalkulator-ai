"""Genetic Programming Symbolic Regression Engine."""

import random
import time
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import sympy as sp

from .expression_tree import BINARY_OPERATORS
from .expression_tree import UNARY_OPERATORS
from .expression_tree import ExpressionTree
from .operators import constant_optimization
from .operators import crossover
from .operators import hoist_mutation
from .operators import point_mutation
from .operators import shrink_mutation
from .pareto_front import ParetoFront
from .pareto_front import ParetoSolution


def huber_loss(y_true, y_pred, delta=1.0):
    """Calculate Huber loss."""
    # Complex-aware Huber loss (using magnitude of error)
    error = y_true - y_pred
    # Clip error magnitude to prevent overflow
    # For complex, we need to handle real/imag parts or just magnitude?
    # Simple magnitude clip:
    # error = np.clip(np.abs(error), 0, 1e100) # Wait, this returns float
    # We need error to stay complex if we were doing gradient descent, but here we just need scalar loss
    
    abs_error = np.abs(error)
    # Clip magnitude
    abs_error = np.clip(abs_error, 0, 1e100)
    
    is_small_error = abs_error <= delta
    squared_loss = 0.5 * abs_error**2
    linear_loss = delta * (abs_error - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)


def weighted_mse(y_true, y_pred, weights=None):
    """Calculate Weighted Mean Squared Error."""
    error = y_true - y_pred
    squared_error = np.abs(error)**2
    
    if weights is not None:
        return np.average(squared_error, weights=weights)
    return np.mean(squared_error)


@dataclass
class GeneticConfig:
    """Configuration for Genetic Symbolic Regression."""

    population_size: int = 500
    n_islands: int = 4
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    parsimony_coefficient: float = 0.01
    max_depth: int = 8
    operators: list[str] = field(
        default_factory=lambda: [
            "add",
            "sub",
            "mul",
            "div",
            "pow",  # Enable a^x patterns like 2^x
            "sin",
            "cos",
            "exp",
            "log",  # Complex-capable log
            "square",
            "sqrt",  # Complex-capable sqrt
            "neg",
            "abs",
        ]
    )
    # Weighted Complexity Config
    # Default weight is 1.0. Higher weights penalize "cheating" operators.
    operator_weights: dict[str, float] = field(
                default_factory=lambda: {
            "max": 5.0,  # Tier 3: Penalize Piecewise Cheating
            "min": 5.0,  # Tier 3
            "abs": 4.0,  # Tier 3: The "Gateway Drug" to max() - heavily penalized
            
            # Tier 2: Physics (Subsidized to match fundamental cost)
            "sin": 1.0,
            "cos": 1.0,
            "tan": 1.0,
            "asin": 1.0,
            "acos": 1.0,
            "atan": 1.0,
            "exp": 1.0,
            "log": 1.0,
            "plog": 1.0,
            "sqrt": 1.0,
            "psqrt": 1.0,
            "pow": 1.0,  # Make 2^x as cheap as 2*x
            
            # Tier 1: Fundamental
            "add": 1.0,
            "sub": 1.0,
            "mul": 1.0,
            "div": 1.0,
        }
    )
    default_complexity_weight: float = 1.0
    
    timeout: float | None = 60.0
    seeds: list[str] = field(default_factory=list)  # Strategy 1: Seeding
    early_stop_mse: float = 1e-10
    verbose: bool = True

    # Advanced options
    constant_optimization_rate: float = 0.1  # Rate of applying constant optimization
    migration_rate: float = 0.1  # Rate of migration between islands
    migration_interval: int = 10  # Generations between migrations
    elitism: int = 5  # Number of best individuals to preserve
    boosting_rounds: int = 1  # Strategy 7: Symbolic Gradient Boosting (1 = off/normal)


class GeneticSymbolicRegressor:
    """Genetic Programming Symbolic Regression Engine.

    Uses evolutionary algorithms to discover mathematical expressions that
    fit the given data. Returns a Pareto front of solutions trading off
    accuracy vs complexity.

    Example:
        >>> regressor = GeneticSymbolicRegressor()
        >>> X = np.linspace(0, 10, 100).reshape(-1, 1)
        >>> y = 3 * X[:, 0]**2 + 2 * X[:, 0] + 1
        >>> pareto = regressor.fit(X, y, variable_names=['x'])
        >>> print(pareto.get_best())
    """

    def __init__(self, config: GeneticConfig | None = None):
        """Initialize the regressor.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or GeneticConfig()
        self.pareto_front: ParetoFront = ParetoFront()
        self.best_tree: ExpressionTree | None = None
        self.generation: int = 0
        self.history: list[dict] = []

        # Filter operators to only valid ones
        self.unary_ops = [op for op in self.config.operators if op in UNARY_OPERATORS]
        self.binary_ops = [op for op in self.config.operators if op in BINARY_OPERATORS]

    def _calculate_smart_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate heuristic importance weights for data points.
        
        Implements the "Vise Strategy" to solve the Parsimony Trap:
        1. Prioritize vertex/origin (x=0) to prevent "lazy" Abs(x) fits.
        2. Prioritize integer anchors (clean numbers) as they are likely ground truth.
        """
        if len(y) == 0:
            return np.ones(0)

        weights = np.ones(len(y), dtype=float)
        
        try:
            # Handle multi-dimensional X
            # Distance from zero (Euclidean norm)
            if X.ndim == 1:
                dist_zero = np.abs(X)
            else:
                dist_zero = np.linalg.norm(X, axis=1) # type: ignore
                
            # 1. VERTEX BONUS (The "Gravity Well")
            # Points closest to 0 get massive boost
            # Find points within reasonable epsilon of 0
            # or just weight by 1/(1+dist)
            
            # Simple approach: Find the single closest point to 0 and boost it 5x
            idx_min = np.argmin(dist_zero)
            # Only if it's actually close (e.g. < 0.1 or normalized?)
            # Let's say if it's the "vertex" representative.
            weights[idx_min] = 5.0
            
            # 2. INTEGER ANCHOR BONUS
            # If x is integer AND y is integer, they are "clean" points.
            # Boost 3x.
            
            # Check X integers (all dims must be integer-ish)
            is_x_int = np.all(np.abs(X - np.round(X)) < 1e-5, axis=1) if X.ndim > 1 else (np.abs(X - np.round(X)) < 1e-5)
            
            # Check Y integers
            is_y_int = (np.abs(y - np.round(y)) < 1e-5)
            
            # Combined mask
            is_anchor = is_x_int & is_y_int
            
            # Apply boost (additive or multiplicative? Multiplicative with base)
            # We set these to 3.0, but if it was already vertex (5.0), keep 5.0?
            # Max rule:
            weights[is_anchor] = np.maximum(weights[is_anchor], 3.0)
            
            # Normalize? No, magnitude matters for "Stiffness" of the loss landscape
            # but for weighted average it cancels out.
            # Wait, np.average(..., weights) normalizes by sum(weights).
            # So relative weights matter.
            
            if self.config.verbose:
                n_anchors = np.sum(is_anchor)
                max_w = np.max(weights)
                if max_w > 1.0:
                    print(f"Smart Weighting: Boosted {n_anchors} anchors (3x) and vertex (5x).")
            
        except Exception as e:
            if self.config.verbose:
                print(f"Smart Weighting failed: {e}. Using uniform weights.")
            return np.ones(len(y))
            
        return weights

    def _calculate_fitness(
        self, 
        tree: ExpressionTree, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: np.ndarray | None = None
    ) -> float:
        """Calculate fitness (MSE + parsimony penalty).

        Args:
            tree: Tree to evaluate
            X: Input data
            y: Target values

        Returns:
            Fitness value (lower is better)
        """
        try:
            # PREVENTION: Skip overly complex expressions that cause SymPy hangs
            # This is more reliable than timeout because Python threads can't be killed
            complexity = tree.complexity(
                weights=self.config.operator_weights, 
                default_weight=self.config.default_complexity_weight
            )
            # Adjust limit for weighted complexity (approx 50 * 1.5 avg weight = 75)
            if complexity > 100:  # Increased limit for weighted schema
                return float("inf")

            # Check expression for patterns that cause SymPy to hang
            expr_str = str(tree.expression) if hasattr(tree, "expression") else ""

            # Too many nested powers
            if expr_str.count("**") > 5:
                return float("inf")

            # Large fractional exponents cause _integer_nthroot_python to hang
            # Match patterns like x**(12345/67890) where numerator > 10000
            import re

            large_frac_pattern = re.compile(
                r"\*\*\s*\(?(-?\d{5,})"
            )  # 5+ digit numbers in power
            if large_frac_pattern.search(expr_str):
                return float("inf")

            # Also check for nested power of power like (x**a)**b
            if "**(" in expr_str and expr_str.count("**") > 2:
                return float("inf")

            # Now safe to evaluate
            predictions = tree.evaluate(X)

            # CRITICAL: Check for NaNs/Infs immediately
            if not np.all(np.isfinite(predictions)):
                return float("inf")

            # Use Huber loss for robustness against outliers
            # This prevents a single outlier from dominating the fitness
            # Use Huber loss for robustness against outliers
            # This prevents a single outlier from dominating the fitness
            # Calculate raw element-wise losss
            raw_loss = huber_loss(y, predictions, delta=1.35)
            
            # Apply weights if provided
            if sample_weight is not None:
                loss = np.average(raw_loss, weights=sample_weight)
            else:
                loss = np.mean(raw_loss)

            # Parsimony pressure: penalize complexity
            # Use Weighted Complexity
            # Coeff might need adjustment if complexity scale changes?
            # 0.01 * 50 = 0.5. 0.01 * 75 = 0.75. Slightly higher penalty is good.
            penalty = self.config.parsimony_coefficient * complexity

            return loss + penalty

        except Exception:
            return float("inf")

    def _calculate_mse(
        self, 
        tree: ExpressionTree, 
        X: np.ndarray, 
        y: np.ndarray,
        sample_weight: np.ndarray | None = None
    ) -> float:
        """Calculate pure MSE without parsimony penalty.

        Args:
            tree: Tree to evaluate
            X: Input data
            y: Target values

        Returns:
            MSE value
        """
        try:
            predictions = tree.evaluate(X)
            
            # CRITICAL: Check for NaNs/Infs immediately
            if not np.all(np.isfinite(predictions)):
                return float("inf")
            # Clip predictions magnitude
            # np.clip on complex numbers only supports real-part clipping in older numpy, 
            # or raises error. Safe way: clip magnitude.
            mag = np.abs(predictions)
            mask = mag > 1e100
            if np.any(mask):
                 scale = np.ones_like(mag)
                 scale[mask] = 1e100 / mag[mask]
                 predictions = predictions * scale

            diff = predictions - y
            
            # Phase-Aware Loss for Complex Log Scaling
            # If we are in log domain, y*log(x) and log(x^y) differ by 2k*pi*i.
            # We want to treat them as equal.
            if hasattr(self, "_log_strategy") and self._log_strategy == "log" and np.iscomplexobj(diff):
                if self._normalization:
                     _, y_scale = self._normalization
                     if y_scale != 0:
                         # Un-normalize imaginary part
                         raw_imag_diff = diff.imag * y_scale
                         
                         # Wrap to [-pi, pi]
                         raw_imag_diff = (raw_imag_diff + np.pi) % (2 * np.pi) - np.pi
                         
                         # Re-normalize
                         new_imag_diff = raw_imag_diff / y_scale
                         
                         # Reconstruct diff
                         diff = diff.real + 1j * new_imag_diff

            # Magnitude squared error
            abs_diff = np.abs(diff)
            np.clip(abs_diff, 0, 1e100, out=abs_diff)
            
            # Magnitude squared error
            abs_diff = np.abs(diff)
            np.clip(abs_diff, 0, 1e100, out=abs_diff)
            
            # Weighted average
            if sample_weight is not None:
                return float(np.average(abs_diff**2, weights=sample_weight))
            return float(np.mean(abs_diff**2))
        except (OverflowError, ValueError, RuntimeWarning):
            return float("inf")
        except Exception:
            return float("inf")

    def _initialize_population(
        self,
        variables: list[str],
        n_individuals: int,
        seeds: list[str] | None = None,
    ) -> list[ExpressionTree]:
        """Initialize a population of random trees.

        Uses ramped half-and-half initialization for diversity.

        Args:
            variables: Variable names
            n_individuals: Number of trees to create
            seeds: Optional list of seed strings (overrides config seeds)

        Returns:
            List of random ExpressionTrees
        """
        population = []

        # Strategy 1: Inject seeds but preserve diversity
        # Limit seeds to at most 50% of population to ensure random diversity
        max_seed_slots = n_individuals // 2
        injected_count = 0

        seeds_to_use = seeds if seeds is not None else self.config.seeds
        
        if seeds_to_use:
            # 1. Parse all valid seeds first
            parsed_trees = []
            for seed_str in seeds_to_use:
                try:
                    import sympy as sp

                    local_dict = {v: sp.Symbol(v) for v in variables}
                    expr = sp.sympify(seed_str, locals=local_dict)
                    tree = ExpressionTree.from_sympy(expr, variables)
                    tree.age = 0
                    parsed_trees.append((seed_str, tree))
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Failed to seed '{seed_str[:50]}...': {e}")
            
            # 2. Distribute slots among seeds
            num_seeds = len(parsed_trees)
            if num_seeds > 0:
                # Calculate how many copies per seed we can afford
                slots_per_seed = max(1, max_seed_slots // num_seeds)
                
                for seed_str, tree in parsed_trees:
                    if len(population) >= max_seed_slots:
                        break

                    # Add original seed
                    population.append(tree)
                    injected_count += 1
                    
                    # Add mutated copies if budget allows
                    copies_needed = slots_per_seed - 1
                    
                    # Cap copies if single seed to avoid dominating with just one idea
                    if num_seeds < 5:
                         copies_needed = min(copies_needed, n_individuals // 5)

                    for i in range(copies_needed):
                        if len(population) >= max_seed_slots:
                            break
                        
                        copy = tree.copy()
                        # Mutate copies to explore neighborhood
                        from .operators import point_mutation
                        
                        # 30% mutation rate for copies
                        copy = point_mutation(
                            copy, mutation_rate=0.3, operators=self.config.operators
                        )
                        copy.age = 0
                        population.append(copy)
                        injected_count += 1

            if self.config.verbose and injected_count > 0:
                print(
                    f"Injected {injected_count} seed expressions (including copies) into population (capped at 50%)"
                )

        # Ramped half-and-half: vary depth and method
        depths = range(2, self.config.max_depth + 1)
        methods = ["grow", "full"]

        # Fill remaining slots
        while len(population) < n_individuals:
            i = len(population)
            depth = depths[i % len(depths)]
            method = methods[i % len(methods)]

            tree = ExpressionTree.random_tree(
                variables=variables,
                max_depth=depth,
                operators=self.config.operators,
                method=method,
            )
            tree.age = 0
            population.append(tree)

        return population

    def _evolve_population(
        self,
        population: list[ExpressionTree],
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
        sample_weight: np.ndarray | None = None,
    ) -> list[ExpressionTree]:
        """Evolve one generation.

        Args:
            population: Current population
            X: Input data
            y: Target values
            generation: Current generation number

        Returns:
            New population
        """
        # Evaluate fitness
        for tree in population:
            if tree.fitness is None or tree.age == 0:
                tree.fitness = self._calculate_fitness(tree, X, y, sample_weight=sample_weight)
            tree.age += 1

        # Sort by fitness (elitism)
        population.sort(key=lambda t: t.fitness)

        new_population = []

        # Elitism
        new_population.extend([t.copy() for t in population[: self.config.elitism]])

        # Selection and breeding
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population)

            if random.random() < self.config.crossover_rate:
                parent2 = self._tournament_select(population)
                offspring1, offspring2 = crossover(parent1, parent2)
                offspring1.age = 0
                new_population.append(offspring1)

                if len(new_population) < self.config.population_size:
                    offspring2.age = 0
                    new_population.append(offspring2)
            else:
                # Mutation (try different types)
                r = random.random()
                parent = parent1.copy()

                if r < 0.7:
                    offspring = point_mutation(
                        parent,
                        self.config.mutation_rate,
                        self.config.operators,
                    )
                elif r < 0.85:
                    offspring = hoist_mutation(parent)
                else:
                    offspring = shrink_mutation(parent)

                offspring.age = 0
                new_population.append(offspring)

        # Occasionally optimize constants (reduced rate to prevent timeout issues)
        if random.random() < 0.02:  # Reduced from 0.1 to prevent long runs
            for _i in range(min(2, len(new_population))):  # Reduced from 5
                idx = random.randrange(len(new_population))
                new_population[idx] = constant_optimization(
                    new_population[idx],
                    X,
                    y,
                    learning_rate=0.1,
                    iterations=2,  # Reduced from 5
                )

        return new_population

    def _tournament_select(self, population: list[ExpressionTree]) -> ExpressionTree:
        """Select best individual from random tournament."""
        tournament = random.sample(population, self.config.tournament_size)
        return min(tournament, key=lambda t: t.fitness)

    def _migrate(self, islands: list[list[ExpressionTree]]):
        """Perform migration between islands.

        Moves best individuals between island populations.

        Args:
            islands: List of island populations
        """
        if len(islands) < 2:
            return

        n_migrants = max(1, int(self.config.migration_rate * len(islands[0])))

        # Ring topology migration
        for i in range(len(islands)):
            source = islands[i]
            target = islands[(i + 1) % len(islands)]

            # Sort source by fitness
            source.sort(key=lambda t: t.fitness)

            # Migrate best individuals
            for j in range(n_migrants):
                if j < len(source):
                    migrant = source[j].copy()
                    migrant.age = 0

                    # Replace worst in target
                    target.sort(key=lambda t: t.fitness)
                    if len(target) > 0:
                        target[-1] = migrant

    def _update_pareto_front(
        self, 
        population: list[ExpressionTree], 
        X: np.ndarray, 
        y: np.ndarray,
        sample_weight: np.ndarray | None = None
    ):
        """Update Pareto front with best solutions from population.

        Args:
            population: Current population
            X: Input data
            y: Target values
        """
        # Only consider top performers to avoid overhead
        sorted_pop = sorted(population, key=lambda t: t.fitness)[:20]

        for tree in sorted_pop:
            mse = self._calculate_mse(tree, X, y, sample_weight=sample_weight)
            if mse < 1e6 and np.isfinite(mse):
                try:
                    # Use simple string for complex trees
                    expr_str = tree.to_pretty_string()

                    # Skip SymPy conversion for complex trees
                    if tree.complexity() > 15:
                        sympy_expr = sp.sympify(0)  # Placeholder
                    else:
                        try:
                            sympy_expr = tree.to_sympy()
                        except Exception:
                            sympy_expr = sp.sympify(0)

                    solution = ParetoSolution(
                        expression=expr_str,
                        sympy_expr=sympy_expr,
                        mse=mse,
                        complexity=tree.complexity(
                            weights=self.config.operator_weights, 
                            default_weight=self.config.default_complexity_weight
                        ),
                        tree=tree.copy(),
                    )
                    self.pareto_front.add(solution)
                except Exception:
                    pass

    def _denormalize_result(self, front: ParetoFront) -> ParetoFront:
        """Denormalize results if normalization was applied."""
        if not self._normalization:
            return front

        y_min, y_scale = self._normalization
        new_front = ParetoFront(max_size=front.max_size)

        for sol in front.solutions:
            try:
                # Transform: raw = norm * scale + min
                # Note: sympy_expr is normalized
                raw_expr = sol.sympy_expr * y_scale + y_min
                
                # Inverse Transform from Log Domain
                if hasattr(self, "_log_strategy") and self._log_strategy:
                    if self._log_strategy == "log":
                         raw_expr = sp.exp(raw_expr)
                    elif self._log_strategy == "arcsinh":
                         raw_expr = sp.sinh(raw_expr)
                    elif self._log_strategy == "linear":
                         pass
                
                # Simplify to clean up constants
                # Simplify to clean up constants
                # E.g. (cosh(x)-min)/scale * scale + min -> cosh(x)
                raw_expr = sp.simplify(raw_expr)
                
                # CLEANUP: Drop negligible additive constants (normalized noise)
                # Since we know y_scale, any constant term significantly smaller than y_scale
                # (e.g. 1e-10 * y_scale) is likely numerical noise from the [0,1] fit.
                
                # Extract constant term
                if raw_expr.is_Add:
                    # Strategy: Separate constant part from variable part
                    # as_coeff_Add returns (constant, rest)
                    coeff, rest = raw_expr.as_coeff_Add()
                    
                    if coeff != 0:
                        # Check magnitude relative to data scale
                        # If constant is < 1e-9 of the data range, drop it
                        # For y_range=80000, this drops constants < 0.00008, which is safe
                        # The observed artifact was 5e-8, which is 6e-13 * scale.
                        # We can be quite aggressive here for "clean" results.
                        
                        relative_mag = abs(float(coeff)) / y_scale
                        if relative_mag < 1e-9:
                           raw_expr = rest
                
                # We need a new tree for the raw expr
                # Use variables from original tree
                variables = sol.tree.variables
                new_tree = ExpressionTree.from_sympy(raw_expr, variables)

                # Recalculate MSE approximate (New MSE = Old MSE * scale^2)
                # Actual MSE might differ slightly due to simplification numericals,
                # but good enough for Pareto ranking preservation.
                new_mse = sol.mse * (y_scale**2)

                new_sol = ParetoSolution(
                    expression=new_tree.to_pretty_string(),
                    sympy_expr=raw_expr,
                    mse=new_mse,
                    complexity=new_tree.complexity(
                        weights=self.config.operator_weights, 
                        default_weight=self.config.default_complexity_weight
                    ),
                    tree=new_tree,
                )
                new_front.add(new_sol)
            except Exception:
                # If conversion fails, fallback is tricky.
                # Assuming simplification usually works.
                pass

        return new_front

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        variable_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None
    ) -> ParetoFront:
        """Fit the symbolic regressor to data (supports Boosting).

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            variable_names: Names of input variables (default: ['x0', 'x1', ...])

        Returns:
        """
        self._sample_weight = sample_weight  # Store for use in internal methods if needed? 
                                             # Actually we pass it down.
        
        # --- SMART WEIGHTING (The "Vise Strategy" Automation) ---
        # If no weights, calculate them heuristically to solve Parsimony Trap
        if sample_weight is None:
            sample_weight = self._calculate_smart_weights(X, y)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        # Ensure correct shape
        if len(y.shape) == 1:
            y = y.flatten()

        # Strategy 7: Symbolic Gradient Boosting Loop
        current_model_tree = None
        y_residual = y.copy()

        interrupted = False
        rounds = self.config.boosting_rounds
        if rounds < 1:
            rounds = 1

        final_front = ParetoFront()

        # Data split strategy (skip if too few samples)
        if len(y) < 5:
            # Not enough data for split/validation, use all for training
            X_train, y_train = X, y
            X_val, y_val = X, y
        else:
            try:
                from sklearn.model_selection import train_test_split

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            except ImportError:
                X_train, y_train = X, y
                X_val, y_val = X, y

        # Normalize y if value range is very large (>1000)
        # This prevents MSE from becoming astronomically large for high-degree polynomials
        # Normalize y if value range is very large (>1000)
        # This prevents MSE from becoming astronomically large for high-degree polynomials
        
        # Handle complex min/max safely
        if np.iscomplexobj(y):
             y_min = 0 # Concept of min/max flawed for complex
             y_max = np.max(np.abs(y)) # Use max magnitude
             y_range = y_max # Range is effectively radius
        else:
             y_min, y_max = y.min(), y.max()
             y_range = y_max - y_min
        self._normalization = None
        self._log_strategy = None  # "log" or "arcsinh"
        
        # Robust Scaling Strategy (Rule 5: No partial fixes)
        # If data spans many orders of magnitude, linear normalization sucks. 
        # Use simple heuristic: if max > 1000 * median (absolute), it's skewed.
        
        y_abs = np.abs(y)
        y_median = np.median(y_abs) if len(y) > 0 else 1.0
        if y_median == 0: y_median = 1e-10 # Prevent div by zero
        skew_ratio = y_max / y_median if y_max > 0 else 0
        
        if skew_ratio > 1000 or (y_range > 1e6):
            # Check for complex data first
            is_complex_y = np.iscomplexobj(y)
            if is_complex_y:
                 # Complex data: TRY LOG FIRST (scimath.log) with PHASE-AWARE LOSS
                 # Linear scaling failed (vanishing gradients).
                 # Log scaling works for range, but needs phase-aware loss to handle branch cuts.
                 self._log_strategy = "log"
                 
                 # Avoid log(0)
                 y_residual_safe = y_residual.copy()
                 y_residual_safe[y_residual == 0] = 1e-10+0j
                 
                 y_residual = np.lib.scimath.log(y_residual_safe)
                 if self.config.verbose:
                    print(f"Data skew detected (ratio {skew_ratio:.1f}). Complex data -> Applying COMPLEX LOG transform.")
            else:
                is_positive = (y_min > 1e-12)
                
                if is_positive:
                    self._log_strategy = "log"
                    y_residual = np.log(y_residual)
                    if self.config.verbose:
                        print(f"Data skew detected (ratio {skew_ratio:.1f}). Inputs positive -> Applying LOG transform.")
                else: 
                     # For mixed real data, log is risky (generates complex from real).
                     # But for x^y with negative base, we WANT complex log.
                     # So let's check if we CAN use complex log?
                     # Actually, let's just use complex log if skew is huge.
                     # If the user gives real data but huge skew involving negatives, 
                     # likely it's a power law with even/odd integers or exp.
                     
                     # FORCE LOG (Complex) for high skew
                     self._log_strategy = "log"
                     
                     # Avoid log(0)
                     y_residual_safe = y_residual.copy()
                     if np.iscomplexobj(y_residual):
                          y_residual_safe[y_residual == 0] = 1e-10+0j
                     else:
                          y_residual_safe[y_residual == 0] = 1e-10
                          
                     y_residual = np.lib.scimath.log(y_residual_safe)
                     if self.config.verbose:
                         print(f"Data skew detected (ratio {skew_ratio:.1f}). Inputs mixed/neg -> Applying COMPLEX LOG transform.")
                    
                     # Fallback to arcsinh if log failed? No scimath.log works.
                     # self._log_strategy = "arcsinh"
                     # y_residual = np.arcsinh(y_residual)

            if self._log_strategy == "log":
                 # Use scimath to handle complex or negative inputs safely
                 
                 # Apply safety to train/val as well
                 y_train_safe = y_train.copy()
                 y_val_safe = y_val.copy()
                 
                 if np.iscomplexobj(y_train):
                     y_train_safe[y_train == 0] = 1e-10+0j
                 else:
                     y_train_safe[y_train == 0] = 1e-10

                 if np.iscomplexobj(y_val):
                     y_val_safe[y_val == 0] = 1e-10+0j
                 else:
                     y_val_safe[y_val == 0] = 1e-10
                     
                 y_train = np.lib.scimath.log(y_train_safe)
                 y_val = np.lib.scimath.log(y_val_safe)
                 
                 y_min_log = 0 
                 y_max_log = np.max(np.abs(y_residual))
                 
                 y_range = y_max_log # Radius
                 y_min = 0 # Scale by max magnitude.
                 
            elif self._log_strategy == "linear":
                 # Fallback if needed, but we prefer log now.
                 pass
                 
            else:
                 y_train = np.arcsinh(y_train)
                 y_val = np.arcsinh(y_val)
                 # Recalculate range
                 # Handle complex min/max for range scaling
                 if is_complex_y:
                      # range is radius
                      y_min_log = 0
                      y_max_log = np.max(np.abs(y_residual))
                 else:
                      y_min_log, y_max_log = np.arcsinh(y_min), np.arcsinh(y_max)
                 
                 y_range = y_max_log - y_min_log
                 y_min = y_min_log

        if y_range > 1000 or self._log_strategy: 
            # Normalize to [0,1] range
            y_scale = y_range + 1e-10
            y_train = (y_train - y_min) / y_scale
            y_val = (y_val - y_min) / y_scale
            y_residual = (y_residual - y_min) / y_scale
            self._normalization = (y_min, y_scale)
            if self.config.verbose:
                print(f"Data normalized ({self._log_strategy or 'linear'}): y range {y_min:.2f} to {y_max_log if self._log_strategy else y_max:.2f} → [0,1]")
        
        # Prepare seeds (Normalize if needed)
        eff_seeds = self.config.seeds
        if self._normalization and eff_seeds:
            y_min, y_scale = self._normalization
            new_seeds = []
            for s in self.config.seeds:
                if self._log_strategy == "log":
                     # log(seed) - min / scale
                     # Need to be careful about seed <= 0. Assuming seeds match data distrib.
                     # Just wrap in log()
                     s_trans = f"(log({s}) - {y_min}) / {y_scale}"
                elif self._log_strategy == "arcsinh":
                     s_trans = f"(asinh({s}) - {y_min}) / {y_scale}"
                else:
                     s_trans = f"(({s}) - {y_min}) / ({y_scale})"
                new_seeds.append(s_trans)
            eff_seeds = new_seeds

        for round_idx in range(rounds):
            if self.config.verbose and rounds > 1:
                print(f"--- Boosting Round {round_idx + 1}/{rounds} ---")

            # Reset state for this round
            self.pareto_front = ParetoFront()
            self.best_tree = None
            self.generation = 0
            self.history = []

            # Initialize islands
            islands = []
            for _ in range(self.config.n_islands):
                island = self._initialize_population(
                    variable_names, self.config.population_size, seeds=eff_seeds
                )
                islands.append(island)

            # Evolution loop
            start_time = time.time()
            if self.config.verbose:
                print(
                    f"Starting evolution with {self.config.n_islands} islands, "
                    f"{self.config.population_size} individuals each..."
                )

            try:
                for gen in range(self.config.generations):
                    self.generation = gen

                    # Check timeout
                    if (
                        self.config.timeout
                        and (time.time() - start_time) > self.config.timeout
                    ):
                        if self.config.verbose:
                            print(f"Timeout after {gen} generations")
                        break

                    # Evolve each island
                    for i, island in enumerate(islands):
                        # Finer timeout check (responsiveness)
                        if (
                            self.config.timeout
                            and (time.time() - start_time) > self.config.timeout
                        ):
                            break

                        islands[i] = self._evolve_population(
                            island, 
                            X_train, 
                            y_train, 
                            gen, 
                            sample_weight=sample_weight if len(y_train)==len(y) else None # Simple split handling
                        ) # Train on Residual

                    # Migration
                    if gen > 0 and gen % self.config.migration_interval == 0:
                        self._migrate(islands)

                    # Update Pareto front
                    for i, island in enumerate(islands):
                        self._update_pareto_front(
                            island, 
                            X_val, 
                            y_val, 
                            sample_weight=sample_weight if len(y_val)==len(y) else None
                        )

                    # Verbose progress output (every 5 generations)
                    if self.config.verbose and gen % 5 == 0:
                        best_res = self.pareto_front.get_best()
                        if best_res:
                            # Truncate expression if too long
                            expr_str = best_res.expression
                            if len(expr_str) > 40:
                                expr_str = expr_str[:37] + "..."
                            print(f"Generation {gen}: Best MSE {best_res.mse:.2e} ({expr_str})")

                    # Early stop check (on Residual)
                    best_res = self.pareto_front.get_best()
                    if best_res and best_res.mse < self.config.early_stop_mse:
                        if self.config.verbose:
                            print(f"Early stop: MSE {best_res.mse:.2e}")
                        break
            except KeyboardInterrupt:
                if self.config.verbose:
                    print("\nEvolution interrupted by user. Stopping current round.")
                interrupted = True

            # End of Round
            # 1. Get best model from this round
            best_round = self.pareto_front.get_best()
            if not best_round:
                if self.config.verbose:
                    print("Warning: No solution found in this round.")
                break

            # 2. Merge into composite model
            if current_model_tree is None:
                current_model_tree = best_round.tree
            else:
                # Merge: F_new = F_old + f_round
                from .expression_tree import ExpressionNode
                from .expression_tree import NodeType

                # Create 'add' node
                root = ExpressionNode(
                    NodeType.BINARY_OP,
                    "add",
                    [
                        current_model_tree.root.copy_subtree(),
                        best_round.tree.root.copy_subtree(),
                    ],
                )
                root.children[0].parent = root
                root.children[1].parent = root

                # Careful: variable names from original context
                current_model_tree = ExpressionTree(root, variable_names)

            # 3. Update residual
            # y_residual = y_original - current_model_pred
            preds = current_model_tree.evaluate(X)
            y_residual = y - preds

            final_mse = np.mean(y_residual**2)
            if final_mse < self.config.early_stop_mse:
                if self.config.verbose:
                    print(f"Boosting converged. Final MSE: {final_mse:.6e}")
                break

            if interrupted:
                break

        # Return final result
        if rounds > 1:
            final_front = ParetoFront()
            if current_model_tree:
                current_model_tree.fitness = self._calculate_fitness(
                    current_model_tree, X, y
                )
                # Create proper solution object
                try:
                    expr_str = current_model_tree.to_pretty_string()
                    try:
                        sympy_expr = current_model_tree.to_sympy()
                    except Exception:
                        sympy_expr = sp.sympify(0)

                    sol = ParetoSolution(
                        expression=expr_str,
                        sympy_expr=sympy_expr,
                        mse=current_model_tree.fitness,  # APPROX
                        complexity=current_model_tree.complexity(),
                        tree=current_model_tree,
                    )
                    final_front.add(sol)
                except Exception:
                    pass
            result_front = self._denormalize_result(final_front)
        else:
            result_front = self._denormalize_result(self.pareto_front)
            
        # Update internal state so predict() works
        self.pareto_front = result_front
        best_sol = self.pareto_front.get_best()
        if best_sol:
            self.best_tree = best_sol.tree
            
        return result_front

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best evolved model.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        if self.best_tree is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.best_tree.evaluate(X)

    def get_expression(self, complexity_budget: int | None = None) -> str:
        """Get the best expression as a string.

        Args:
            complexity_budget: Maximum allowed complexity

        Returns:
            String representation of the best expression
        """
        solution = self.pareto_front.get_best(complexity_budget)
        if solution:
            return solution.expression
        return ""

    def get_sympy(self, complexity_budget: int | None = None) -> sp.Expr:
        """Get the best expression as a SymPy object.

        Args:
            complexity_budget: Maximum allowed complexity

        Returns:
            SymPy expression
        """
        solution = self.pareto_front.get_best(complexity_budget)
        if solution:
            return solution.sympy_expr
        return sp.Integer(0)


def discover_equation(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str] | None = None,
    timeout: float = 30.0,
    verbose: bool = True,
) -> tuple[str, float, ParetoFront]:
    """Convenience function to discover an equation from data.

    Args:
        X: Input data
        y: Target values
        variable_names: Variable names
        timeout: Maximum seconds to run
        verbose: Print progress

    Returns:
        Tuple of (best expression string, MSE, full Pareto front)
    """
    config = GeneticConfig(
        population_size=200,
        n_islands=2,
        generations=100,
        timeout=timeout,
        verbose=verbose,
    )

    regressor = GeneticSymbolicRegressor(config)
    pareto = regressor.fit(X, y, variable_names)

    best = pareto.get_knee_point() or pareto.get_best()
    if best:
        return best.expression, best.mse, pareto
    return "", float("inf"), pareto
