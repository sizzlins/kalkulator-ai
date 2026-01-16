"""Strategies for Genetic Symbolic Regression."""
import numpy as np
import random
import time
from .genetic_config import GeneticConfig
from .expression_tree import ExpressionTree, UNARY_OPERATORS, BINARY_OPERATORS
from .operators import (
    crossover, point_mutation, hoist_mutation, shrink_mutation, 
    constant_optimization
)
from .constant_anchors import detect_anchors, generate_hypotheses

class BoostingStrategy:
    """Manages data weighting and boosting logic."""
    
    def __init__(self, config: GeneticConfig):
        self.config = config

    def calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate heuristic importance weights (The "Vise Strategy")."""
        if len(y) == 0:
            return np.ones(0)

        weights = np.ones(len(y), dtype=float)
        
        try:
            # 1. Distance from zero (Euclidean norm)
            if X.ndim == 1:
                dist_zero = np.abs(X)
            else:
                dist_zero = np.linalg.norm(X, axis=1)
                
            # VERTEX BONUS: Boost point closest to 0
            idx_min = np.argmin(dist_zero)
            weights[idx_min] = self.config.vertex_bonus
            
            # 2. INTEGER ANCHOR BONUS
            # Check X integers (using configurable tolerance)
            tol = self.config.integer_tolerance
            is_x_int = np.all(np.abs(X - np.round(X)) < tol, axis=1) if X.ndim > 1 else (np.abs(X - np.round(X)) < tol)
            # Check Y integers
            is_y_int = (np.abs(y - np.round(y)) < tol)
            
            is_anchor = is_x_int & is_y_int
            weights[is_anchor] = np.maximum(weights[is_anchor], self.config.anchor_bonus)
            
            if self.config.verbose:
                n_anchors = np.sum(is_anchor)
                max_w = np.max(weights)
                if max_w > 1.0:
                    print(f"Smart Weighting: Boosted {n_anchors} anchors and vertex.")
            
        except (ValueError, IndexError, TypeError) as e:
            if self.config.verbose:
                print(f"Smart Weighting failed: {e}. Using uniform weights.")
            return np.ones(len(y))
            
        return weights


class EvolutionStrategy:
    """Manages the evolutionary lifecycle of the population."""
    
    def __init__(self, config: GeneticConfig):
        self.config = config
        
    def huber_loss(self, y_true, y_pred, delta=1.35):
        """Robust loss function."""
        error = y_true - y_pred
        abs_error = np.abs(error)
        abs_error = np.clip(abs_error, 0, 1e100)
        
        is_small_error = abs_error <= delta
        squared_loss = 0.5 * abs_error**2
        linear_loss = delta * (abs_error - 0.5 * delta)
        return np.where(is_small_error, squared_loss, linear_loss)

    def calculate_fitness(self, tree: ExpressionTree, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        """Evaluate tree fitness (Loss + Parsimony)."""
        try:
            # 1. Complexity Check (Fail fast)
            complexity = tree.complexity(
                weights=self.config.operator_weights, 
                default_weight=self.config.default_complexity_weight
            )
            if complexity > self.config.complexity_limit:
                return float("inf")

            expr_str = str(tree)
            if expr_str.count("**") > self.config.max_nested_powers:
                return float("inf")

            # 2. Evaluation
            predictions = tree.evaluate(X)
            if not np.all(np.isfinite(predictions)):
                return float("inf")

            # 3. Loss Calculation
            raw_loss = self.huber_loss(y, predictions)
            
            if sample_weight is not None:
                loss = np.average(raw_loss, weights=sample_weight)
            else:
                loss = np.mean(raw_loss)

            # Cache raw MSE for Pareto updates (avoids recalculation)
            tree._cached_mse = loss

            # 4. Perfect Fit Bypass (No penalty if perfect)
            if loss < self.config.early_stop_mse:
                return loss

            return loss + (self.config.parsimony_coefficient * complexity)

        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            return float("inf")

    def initialize_population(self, variables: list[str], n_individuals: int, seeds: list[str] = None, X=None, y=None) -> list[ExpressionTree]:
        """Create initial population with seeds and random trees."""
        population = []
        
        # 1. Anchor Detection / Seeding
        anchor_seeds = []
        if X is not None and y is not None:
            try:
                anchors = detect_anchors(X, y)
                if anchors:
                    anchor_seeds = generate_hypotheses(anchors, variables[0])
            except (ValueError, TypeError, IndexError):
                pass

        combined_seeds = (seeds or []) + anchor_seeds
        
        # 2. Inject Seeds
        if combined_seeds:
            import sympy as sp
            local_dict = {v: sp.Symbol(v) for v in variables}
            # Add safe globals for parsing
            local_dict.update({'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos})
            
            for seed_str in combined_seeds[:n_individuals // 2]:
                try:
                    expr = sp.sympify(seed_str, locals=local_dict)
                    tree = ExpressionTree.from_sympy(expr, variables)
                    tree.age = 0
                    population.append(tree)
                except (ValueError, TypeError, SyntaxError, AttributeError):
                    pass

        # 3. Fill remaining with Random Trees (Ramped Half-and-Half)
        depths = range(2, self.config.max_depth + 1)
        methods = ["grow", "full"]
        
        while len(population) < n_individuals:
            depth = depths[len(population) % len(depths)]
            method = methods[len(population) % len(methods)]
            tree = ExpressionTree.random_tree(
                variables=variables,
                max_depth=depth,
                operators=self.config.operators,
                method=method
            )
            tree.age = 0
            population.append(tree)
            
        return population

    def evolve(self, population: list[ExpressionTree], X, y, generation, sample_weight=None) -> list[ExpressionTree]:
        """Run one generation of evolution."""
        # 1. Evaluate
        for tree in population:
            if tree.fitness is None or tree.age == 0:
                tree.fitness = self.calculate_fitness(tree, X, y, sample_weight)
            tree.age += 1
            
        # 2. Elitism
        population.sort(key=lambda t: t.fitness)
        new_pop = [t.copy() for t in population[:self.config.elitism]]
        
        # 3. Breeding
        while len(new_pop) < self.config.population_size:
            parent1 = self._tournament_select(population)
            
            if random.random() < self.config.crossover_rate:
                parent2 = self._tournament_select(population)
                off1, off2 = crossover(parent1, parent2)
                off1.age = 0; off2.age = 0
                new_pop.append(off1)
                if len(new_pop) < self.config.population_size:
                    new_pop.append(off2)
            else:
                # Mutation
                r = random.random()
                parent = parent1.copy()
                if r < 0.7:
                    child = point_mutation(parent, self.config.mutation_rate, self.config.operators)
                elif r < 0.85:
                    child = hoist_mutation(parent)
                else:
                    child = shrink_mutation(parent)
                child.age = 0
                new_pop.append(child)
                
        # 4. Constant Optimization (Stochastic Hill Climbing)
        if random.random() < self.config.constant_optimization_rate:
            idx = random.randrange(len(new_pop))
            new_pop[idx] = constant_optimization(
                new_pop[idx], X, y, learning_rate=0.1, iterations=2
            )
            
        return new_pop

    def _tournament_select(self, population):
        sample = random.sample(population, min(len(population), self.config.tournament_size))
        return min(sample, key=lambda t: t.fitness)

    def migrate(self, islands: list[list[ExpressionTree]]):
        """Ring topology migration."""
        if len(islands) < 2: return
        n_migrants = max(1, int(self.config.migration_rate * len(islands[0])))
        
        for i in range(len(islands)):
            source = islands[i]
            target = islands[(i + 1) % len(islands)]
            
            source.sort(key=lambda t: t.fitness)
            target.sort(key=lambda t: t.fitness)
            
            # Swap best of source with worst of target
            for j in range(n_migrants):
                if j < len(source) and len(target) > 0:
                    target[-(j+1)] = source[j].copy()
