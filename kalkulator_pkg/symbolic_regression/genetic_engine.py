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
    error = y_true - y_pred
    # Clip error to prevent overflow in square
    error = np.clip(error, -1e100, 1e100)
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()


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
            "log",
            "square",
            "sqrt",
            "neg",
            "abs",
            "max",
            "min",
        ]
    )
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

    def _calculate_fitness(
        self, tree: ExpressionTree, X: np.ndarray, y: np.ndarray
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
            complexity = tree.complexity()
            if complexity > 50:  # Very complex expressions often cause hangs
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

            # Use Huber loss for robustness against outliers
            # This prevents a single outlier from dominating the fitness
            loss = huber_loss(y, predictions, delta=1.35)

            # Parsimony pressure: penalize complexity
            penalty = self.config.parsimony_coefficient * complexity

            return loss + penalty

        except Exception:
            return float("inf")

    def _calculate_mse(
        self, tree: ExpressionTree, X: np.ndarray, y: np.ndarray
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
            # Clip predictions to avoid overflow in square
            # Use float64 max sqrt approx 1e150, but let's be safer 1e100
            np.clip(predictions, -1e100, 1e100, out=predictions)

            diff = predictions - y
            # Further protection against squaring large diffs
            np.clip(diff, -1e100, 1e100, out=diff)

            return float(np.mean(diff**2))
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

        # Strategy 1: Inject seeds with multiple copies for survival
        # Single seeds get overwhelmed by random population - inject ~10% as seed copies
        injected_count = 0
        seed_copies_target = max(
            10, n_individuals // 10
        )  # At least 10, or 10% of population

        seeds_to_use = seeds if seeds is not None else self.config.seeds
        if seeds_to_use:
            for seed_str in seeds_to_use:
                try:
                    import sympy as sp

                    local_dict = {v: sp.Symbol(v) for v in variables}
                    expr = sp.sympify(seed_str, locals=local_dict)
                    tree = ExpressionTree.from_sympy(expr, variables)
                    tree.age = 0

                    # Add original seed
                    population.append(tree)
                    injected_count += 1

                    # Add mutated copies to fill ~10% of population
                    copies_to_add = min(
                        seed_copies_target - 1, n_individuals - len(population)
                    )
                    unmutated_count = max(1, copies_to_add // 5)  # Keep 20% unmutated

                    for i in range(copies_to_add):
                        copy = tree.copy()
                        # First 20% of copies: keep exact (no mutation)
                        # Remaining 80%: mutate for diversity
                        if i >= unmutated_count:
                            from .operators import point_mutation

                            copy = point_mutation(
                                copy, mutation_rate=0.3, operators=self.config.operators
                            )
                        copy.age = 0
                        population.append(copy)
                        injected_count += 1

                except Exception as e:
                    if self.config.verbose:
                        print(f"Warning: Failed to seed '{seed_str[:50]}...': {e}")

            if self.config.verbose and injected_count > 0:
                print(
                    f"Injected {injected_count} seed expressions (including copies) into population"
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
                tree.fitness = self._calculate_fitness(tree, X, y)
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
        self, population: list[ExpressionTree], X: np.ndarray, y: np.ndarray
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
            mse = self._calculate_mse(tree, X, y)
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
                        complexity=tree.complexity(),
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
                # Simplify to clean up constants
                # E.g. (cosh(x)-min)/scale * scale + min -> cosh(x)
                raw_expr = sp.simplify(raw_expr)

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
                    complexity=new_tree.complexity(),
                    tree=new_tree,
                )
                new_front.add(new_sol)
            except Exception:
                # If conversion fails, fallback is tricky.
                # Assuming simplification usually works.
                pass

        return new_front

    def fit(
        self, X: np.ndarray, y: np.ndarray, variable_names: list[str] | None = None
    ) -> ParetoFront:
        """Fit the symbolic regressor to data (supports Boosting).

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            variable_names: Names of input variables (default: ['x0', 'x1', ...])

        Returns:
            Pareto front of solutions
        """
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        # Ensure correct shape
        if len(y.shape) == 1:
            y = y.flatten()

        # Strategy 7: Symbolic Gradient Boosting Loop
        current_model_tree = None
        y_residual = y.copy()

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
                _X_train, y_train = X, y
                _X_val, y_val = X, y

        # Normalize y if value range is very large (>1000)
        # This prevents MSE from becoming astronomically large for high-degree polynomials
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        self._normalization = None

        if y_range > 1000:
            # Normalize to [0,1] range
            y_scale = y_range + 1e-10
            y_train = (y_train - y_min) / y_scale
            y_val = (y_val - y_min) / y_scale
            y_residual = (y_residual - y_min) / y_scale
            self._normalization = (y_min, y_scale)
            if self.config.verbose:
                print(f"Data normalized: y range {y_min:.2f} to {y_max:.2f} → [0,1]")
        
        # Prepare seeds (Normalize if needed)
        eff_seeds = self.config.seeds
        if self._normalization and eff_seeds:
            y_min, y_scale = self._normalization
            # Transform seeds: (seed - min) / scale
            # Wrap in parens to ensure precedence
            eff_seeds = [
                f"(({s}) - {y_min}) / ({y_scale})" for s in self.config.seeds
            ]

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
                    islands[i] = self._evolve_population(
                        island, X, y_residual, gen
                    )  # Train on Residual

                # Migration
                if gen > 0 and gen % self.config.migration_interval == 0:
                    self._migrate(islands)

                # Update Pareto front
                for island in islands:
                    self._update_pareto_front(island, X, y_residual)

                # Early stop check (on Residual)
                best_res = self.pareto_front.get_best()
                if best_res and best_res.mse < self.config.early_stop_mse:
                    if self.config.verbose:
                        print(f"Early stop: MSE {best_res.mse:.2e}")
                    break

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
