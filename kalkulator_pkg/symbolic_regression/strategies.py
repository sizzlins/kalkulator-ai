"""Strategies for Genetic Symbolic Regression."""
import numpy as np
import random
import time
from .genetic_config import GeneticConfig
from .expression_tree import ExpressionTree, UNARY_OPERATORS, BINARY_OPERATORS
from .operators import (
    crossover, point_mutation, hoist_mutation, shrink_mutation, 
    optimize_constants_bfgs
)
try:
    from .nsga2 import assign_nsga2_ranks, tournament_select_ranked
except ImportError:
    # Fallback if nsga2 module issues
    def assign_nsga2_ranks(pop): pass
    def tournament_select_ranked(pop, size): return random.choice(pop)

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
            if self.config.use_integer_anchoring:
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

    def evaluate_population(self, population: list[ExpressionTree], X: np.ndarray, y: np.ndarray, sample_weight=None):
        """Evaluate fitness for the entire population."""
        for tree in population:
            if tree.fitness is None or tree.age == 0:
                tree.fitness = self.calculate_fitness(tree, X, y, sample_weight)
            # tree.age += 1 # Age increment happens in evolve() typically
        
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
        # Ignore all warnings (don't crash on div by zero, etc.)
        with np.errstate(all='ignore'):
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

                # 2. Evaluation (using fast compiled path with fallback)
                predictions = tree.evaluate_fast(X)
                
                # Check for garbage (NaN, Inf)
                # Note: Complex values ARE allowed (handled by np.abs in loss function)
                if not np.all(np.isfinite(predictions)):
                    return float("inf")
                
                # REMOVED: if np.iscomplexobj(predictions): return float("inf")
                # We want to support complex regression (e.g. LambertW, exponents)

                # 3. Loss Calculation
                # CRITICAL FIX: Ensure predictions and y are both flattened 1D arrays
                # to prevent accidental (N,N) matrix broadcasting if one is (N,1)
                y_flat = np.asarray(y).flatten()
                pred_flat = np.asarray(predictions).flatten()
                
                raw_loss = self.huber_loss(y_flat, pred_flat)
                
                if sample_weight is not None:
                    # Ensure shapes match for weighted average
                    loss = np.average(raw_loss, weights=np.asarray(sample_weight).flatten())
                else:
                    loss = np.mean(raw_loss)

                # Cache raw MSE for Pareto updates (avoids recalculation)
                tree._cached_mse = loss

                # 4. Perfect Fit Bypass (No penalty if perfect)
                # 4. Perfect Fit Bypass (No penalty if perfect)
                if loss < self.config.early_stop_mse:
                    return loss

                # 5. Integer Penalty (The "Race Problem" Fix)
                # If target is purely integer but prediction is floaty, penalize heavily.
                # This prevents 0.49*x from beating x >> 1.
                try:
                    target_is_int = np.all(np.equal(np.mod(y_flat, 1), 0))
                    if target_is_int:
                        # Check if prediction is "dirty" (not close to integers)
                        # Tolerance 1e-6 allows for float math 5.000001
                        pred_is_dirty = not np.all(np.abs(pred_flat - np.round(pred_flat)) < 1e-6)
                        
                        if pred_is_dirty:
                            # Massive penalty to discourage float approximations for integer problems
                            loss += 10.0
                except Exception:
                    # Robustness: if checking fails (e.g. types), skip penalty
                    pass

                return loss + (self.config.parsimony_coefficient * complexity)

            except Exception:
                # Catch-All for any runtime error (ZeroDivision, Overflow, etc.)
                # This prevents worker process termination ("Poison Pill").
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
            # SECURITY: Import safe parser instead of using sympify (which uses eval)
            from ..parser import safe_sympy_parse
            from ..config import ALLOWED_SYMPY_NAMES
            
            local_dict = {v: sp.Symbol(v) for v in variables}
            # Add safe globals for parsing
            local_dict.update({'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos})
            # Merge with allowed names for comprehensive coverage
            full_local_dict = {**ALLOWED_SYMPY_NAMES, **local_dict}
            
            for seed_str in combined_seeds[:n_individuals // 2]:
                try:
                    # SECURITY: Use AST-based safe parser instead of eval-based sympify
                    expr = safe_sympy_parse(seed_str, local_dict=full_local_dict)
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

    def evolve(self, population: list[ExpressionTree], X: np.ndarray, y: np.ndarray, generation: int, sample_weight=None) -> list[ExpressionTree]:
        """Run one generation of evolution."""
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Entering evolve...")
        
        # 1. Evaluate
        for i, tree in enumerate(population):
            if tree.fitness is None or tree.age == 0:
                tree.fitness = self.calculate_fitness(tree, X, y, sample_weight)
            # tree.age += 1
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Evaluation done.")
            
        # 2. Elitism
        population.sort(key=lambda t: t.fitness)
        new_pop = [t.copy() for t in population[:self.config.elitism]]
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Elitism done.")
        
        # 3. Assign NSGA-II Ranks (for selection)
        assign_nsga2_ranks(population)
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] NSGA-II Ranks done.")
        
        # 4. Breeding
        while len(new_pop) < self.config.population_size:
            parent1 = self._tournament_select(population)
            
            if random.random() < self.config.crossover_rate:
                parent2 = self._tournament_select(population)
                off1, off2 = crossover(parent1, parent2)
                off1.age = 0; off2.age = 0
                # CRITICAL: Reset fitness to prevent zombies
                off1.fitness = None; off2.fitness = None
                new_pop.append(off1)
                if len(new_pop) < self.config.population_size:
                    new_pop.append(off2)
            else:
                # Mutation
                r = random.random()
                parent = parent1.copy()
                # 60% Point, 10% Hoist, 10% Shrink, 20% Insert (Encourage nesting for complex funcs)
                if r < 0.6:
                    child = point_mutation(parent, self.config.mutation_rate, self.config.operators)
                elif r < 0.8:  # 20% chance to insert (wrap)
                    from .operators import insert_mutation
                    child = insert_mutation(parent, self.config.operators)
                elif r < 0.9:
                    child = hoist_mutation(parent)
                else:
                    child = shrink_mutation(parent)
                child.age = 0
                # CRITICAL: Reset fitness to prevent zombies
                child.fitness = None
                new_pop.append(child)
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Breeding done.")

        # 5. Constant Optimization (BFGS - fast gradient-based)
        if random.random() < self.config.constant_optimization_rate:
            # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Starting BFGS...")
            idx = random.randrange(len(new_pop))
            new_pop[idx] = optimize_constants_bfgs(
                new_pop[idx], X, y, max_iter=10
            )
            # if self.config.verbose: print(f"DEBUG: [Gen {generation}] BFGS done.")
            
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Exiting evolve.")
        return new_pop

    def _tournament_select(self, population):
        """Select parent using NSGA-II ranking."""
        return tournament_select_ranked(population, self.config.tournament_size)

    def migrate(self, islands: list[list[ExpressionTree]]):
        """Ring topology migration."""
        if len(islands) < 2: return
        n_migrants = max(1, int(self.config.migration_rate * len(islands[0])))
        
        for i in range(len(islands)):
            source = islands[i]
            target = islands[(i + 1) % len(islands)]
            
            # CRITICAL FIX: Handle None fitness (from unevaluated offspring)
            # Treat None as infinity (worst possible fitness)
            source.sort(key=lambda t: t.fitness if t.fitness is not None else float('inf'))
            target.sort(key=lambda t: t.fitness if t.fitness is not None else float('inf'))
            
            # Swap best of source with worst of target
            for j in range(n_migrants):
                if j < len(source) and len(target) > 0:
                    target[-(j+1)] = source[j].copy()
