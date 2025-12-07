"""Genetic Programming Symbolic Regression Engine.

This is the main evolutionary algorithm for discovering mathematical expressions
from data. It evolves a population of expression trees, selecting for both
accuracy (low MSE) and simplicity (low complexity).

Key Features:
- Multi-population island model for diversity
- Pareto optimization for accuracy vs complexity trade-off
- Age-based population management to prevent premature convergence
- Automatic constant optimization
"""

from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

from .expression_tree import ExpressionTree, UNARY_OPERATORS, BINARY_OPERATORS
from .operators import (
    point_mutation,
    subtree_mutation,
    hoist_mutation,
    shrink_mutation,
    constant_optimization,
    crossover,
    tournament_selection,
    lexicographic_selection,
)
from .pareto_front import ParetoFront, ParetoSolution


@dataclass
class GeneticConfig:
    """Configuration for the genetic symbolic regression algorithm.
    
    Attributes:
        population_size: Number of individuals per island
        n_islands: Number of parallel populations
        generations: Maximum generations to evolve
        tournament_size: Size of selection tournaments
        crossover_rate: Probability of crossover vs mutation
        mutation_rate: Probability per node for point mutation
        parsimony_coefficient: Penalty per complexity unit
        max_depth: Maximum allowed tree depth
        operators: List of allowed operators
        timeout: Maximum seconds to run (None for no limit)
        early_stop_mse: Stop if MSE drops below this threshold
        verbose: Print progress information
    """
    population_size: int = 500
    n_islands: int = 4
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    parsimony_coefficient: float = 0.001
    max_depth: int = 8
    operators: list[str] = field(default_factory=lambda: [
        'add', 'sub', 'mul', 'div',
        'sin', 'cos', 'exp', 'log',
        'square', 'sqrt', 'neg', 'abs'
    ])
    timeout: float | None = 60.0
    early_stop_mse: float = 1e-10
    verbose: bool = True
    
    # Advanced options
    constant_optimization_rate: float = 0.1  # Rate of applying constant optimization
    migration_rate: float = 0.1  # Rate of migration between islands
    migration_interval: int = 10  # Generations between migrations
    elitism: int = 5  # Number of best individuals to preserve


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
        self,
        tree: ExpressionTree,
        X: np.ndarray,
        y: np.ndarray
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
            predictions = tree.evaluate(X)
            mse = np.mean((predictions - y) ** 2)
            
            # Parsimony pressure: penalize complexity
            penalty = self.config.parsimony_coefficient * tree.complexity()
            
            return mse + penalty
            
        except Exception:
            return float('inf')
    
    def _calculate_mse(
        self,
        tree: ExpressionTree,
        X: np.ndarray,
        y: np.ndarray
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
            return float(np.mean((predictions - y) ** 2))
        except Exception:
            return float('inf')
    
    def _initialize_population(
        self,
        variables: list[str],
        n_individuals: int
    ) -> list[ExpressionTree]:
        """Initialize a population of random trees.
        
        Uses ramped half-and-half initialization for diversity.
        
        Args:
            variables: Variable names
            n_individuals: Number of trees to create
            
        Returns:
            List of random ExpressionTrees
        """
        population = []
        
        # Ramped half-and-half: vary depth and method
        depths = range(2, self.config.max_depth + 1)
        methods = ['grow', 'full']
        
        for i in range(n_individuals):
            depth = depths[i % len(depths)]
            method = methods[i % len(methods)]
            
            tree = ExpressionTree.random_tree(
                variables=variables,
                max_depth=depth,
                operators=self.config.operators,
                method=method
            )
            tree.age = 0
            population.append(tree)
        
        return population
    
    def _evolve_population(
        self,
        population: list[ExpressionTree],
        X: np.ndarray,
        y: np.ndarray,
        generation: int
    ) -> list[ExpressionTree]:
        """Evolve population for one generation.
        
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
            if tree.fitness == float('inf'):
                tree.fitness = self._calculate_fitness(tree, X, y)
        
        # Sort by fitness
        population.sort(key=lambda t: t.fitness)
        
        # New population
        new_population = []
        
        # Elitism: keep best individuals
        elite = population[:self.config.elitism]
        for tree in elite:
            new_tree = tree.copy()
            new_tree.age = tree.age + 1
            new_population.append(new_tree)
        
        # Fill rest with offspring
        while len(new_population) < len(population):
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = tournament_selection(population, self.config.tournament_size)
                parent2 = tournament_selection(population, self.config.tournament_size)
                offspring1, offspring2 = crossover(parent1, parent2, self.config.max_depth)
                offspring1.age = 0
                offspring2.age = 0
                new_population.append(offspring1)
                if len(new_population) < len(population):
                    new_population.append(offspring2)
            else:
                # Mutation
                parent = tournament_selection(population, self.config.tournament_size)
                
                # Choose mutation type
                r = random.random()
                if r < 0.4:
                    offspring = point_mutation(parent, self.config.mutation_rate, self.config.operators)
                elif r < 0.7:
                    offspring = subtree_mutation(parent, max_depth=3, operators=self.config.operators)
                elif r < 0.85:
                    offspring = hoist_mutation(parent)
                else:
                    offspring = shrink_mutation(parent)
                
                offspring.age = 0
                new_population.append(offspring)
        
        # Occasionally optimize constants
        if random.random() < self.config.constant_optimization_rate:
            for i in range(min(5, len(new_population))):
                idx = random.randrange(len(new_population))
                new_population[idx] = constant_optimization(
                    new_population[idx], X, y,
                    learning_rate=0.1, iterations=5
                )
        
        return new_population
    
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
        y: np.ndarray
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
                        tree=tree.copy()
                    )
                    self.pareto_front.add(solution)
                except Exception:
                    pass
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: list[str] | None = None
    ) -> ParetoFront:
        """Fit the symbolic regressor to data.
        
        Args:
            X: Input data of shape (n_samples,) or (n_samples, n_features)
            y: Target values of shape (n_samples,)
            variable_names: Names for input variables (default: x, y, z, ...)
            
        Returns:
            ParetoFront containing Pareto-optimal solutions
        """
        # Prepare data
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Default variable names
        if variable_names is None:
            if n_features == 1:
                variable_names = ['x']
            else:
                variable_names = [chr(ord('x') + i) if i < 3 else f'x{i}' 
                                  for i in range(n_features)]
        
        # Initialize islands
        islands = []
        for _ in range(self.config.n_islands):
            island = self._initialize_population(
                variable_names,
                self.config.population_size
            )
            islands.append(island)
        
        # Evolution loop
        start_time = time.time()
        self.generation = 0
        
        if self.config.verbose:
            print(f"Starting evolution with {self.config.n_islands} islands, "
                  f"{self.config.population_size} individuals each...")
        
        for gen in range(self.config.generations):
            self.generation = gen
            
            # Check timeout
            if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                if self.config.verbose:
                    print(f"Timeout after {gen} generations")
                break
            
            # Evolve each island
            for i, island in enumerate(islands):
                islands[i] = self._evolve_population(island, X, y, gen)
            
            # Migration
            if gen > 0 and gen % self.config.migration_interval == 0:
                self._migrate(islands)
            
            # Update Pareto front
            for island in islands:
                self._update_pareto_front(island, X, y)
            
            # Track best
            all_trees = [t for island in islands for t in island]
            best_current = min(all_trees, key=lambda t: t.fitness)
            best_mse = self._calculate_mse(best_current, X, y)
            
            # Store history
            self.history.append({
                'generation': gen,
                'best_mse': best_mse,
                'best_complexity': best_current.complexity(),
                'pareto_size': len(self.pareto_front),
            })
            
            if self.config.verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best MSE = {best_mse:.6e}, "
                      f"Complexity = {best_current.complexity()}, "
                      f"Pareto size = {len(self.pareto_front)}")
            
            # Early stopping
            if best_mse < self.config.early_stop_mse:
                if self.config.verbose:
                    print(f"Early stop at generation {gen}: MSE = {best_mse:.2e}")
                break
        
        # Final update
        for island in islands:
            self._update_pareto_front(island, X, y)
        
        # Store best tree
        best_solution = self.pareto_front.get_best()
        if best_solution:
            self.best_tree = best_solution.tree
        
        if self.config.verbose:
            elapsed = time.time() - start_time
            print(f"\nEvolution complete in {elapsed:.1f}s")
            print(f"Pareto front contains {len(self.pareto_front)} solutions")
            if best_solution:
                print(f"Best: {best_solution.expression}")
                print(f"  MSE: {best_solution.mse:.6e}")
                print(f"  Complexity: {best_solution.complexity}")
        
        return self.pareto_front
    
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
    verbose: bool = True
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
        verbose=verbose
    )
    
    regressor = GeneticSymbolicRegressor(config)
    pareto = regressor.fit(X, y, variable_names)
    
    best = pareto.get_knee_point() or pareto.get_best()
    if best:
        return best.expression, best.mse, pareto
    return "", float('inf'), pareto
