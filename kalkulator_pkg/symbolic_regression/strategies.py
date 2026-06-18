"""Strategies for Genetic Symbolic Regression."""
import numpy as np
import random
from .genetic_config import GeneticConfig
from .expression_tree import ExpressionTree
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
        # Opt 1: Fitness cache keyed by RPN tuple (avoids re-evaluating identical trees)
        self._fitness_cache: dict[tuple, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        # Opt 3: Fixed random probe indices (set on first evaluate call)
        self._probe_indices: np.ndarray | None = None
    
    def reset_cache(self):
        """Clear fitness cache. Call between boosting rounds when target y changes."""
        if self.config.verbose and self._cache_hits + self._cache_misses > 0:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total * 100 if total > 0 else 0
            print(f"[Cache] Reset. Hit rate: {self._cache_hits}/{total} ({hit_rate:.0f}%)")
        self._fitness_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._probe_indices = None  # Reset probe for new data

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
        """Evaluate tree fitness (Loss + Parsimony).
        
        Performance optimizations:
        - RPN tuple cache: skip evaluation for structurally identical trees
        - Safety probe: reject garbage trees early with a 20-point subset
        """
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

                # Opt 1: Fitness Cache — RPN tuple key
                try:
                    cache_key = tuple(tree.root.to_rpn())
                except Exception:
                    cache_key = None
                
                if cache_key is not None and cache_key in self._fitness_cache:
                    self._cache_hits += 1
                    cached = self._fitness_cache[cache_key]
                    tree._cached_mse = cached
                    return cached
                if cache_key is not None:
                    self._cache_misses += 1

                # Opt 3: Safety Probe — reject garbage trees early
                if self._probe_indices is None and len(X) > 40:
                    self._probe_indices = np.random.choice(
                        len(X), size=min(20, len(X)), replace=False
                    )
                
                if self._probe_indices is not None:
                    probe_pred = tree.evaluate_fast(X[self._probe_indices])
                    if not np.all(np.isfinite(probe_pred)):
                        if cache_key is not None:
                            self._fitness_cache[cache_key] = float('inf')
                        return float('inf')

                # 2. Full Evaluation (using fast compiled path with fallback)
                predictions = tree.evaluate_fast(X)
                
                # Check for garbage (NaN, Inf)
                if not np.all(np.isfinite(predictions)):
                    if cache_key is not None:
                        self._fitness_cache[cache_key] = float('inf')
                    return float("inf")

                # 3. Loss Calculation
                y_flat = np.asarray(y).flatten()
                pred_flat = np.asarray(predictions).flatten()
                
                if getattr(self.config, 'loss_function', 'huber') == 'pearson':
                    # Pearson Correlation Loss
                    # We want to maximize correlation |r|, so loss = 1.0 - |r|
                    # Add small epsilon to avoid divide by zero if variance is zero
                    y_std = np.std(y_flat)
                    pred_std = np.std(pred_flat)
                    
                    if y_std < 1e-10 or pred_std < 1e-10:
                        # Flat line -> terrible correlation loss
                        loss = 1.0
                    else:
                        cov = np.cov(y_flat, pred_flat)[0, 1]
                        r = cov / (y_std * pred_std)
                        # r can be NaN if floating point issues, fallback to 1.0 loss
                        if np.isnan(r):
                            loss = 1.0
                        else:
                            # We take absolute value because a negative constant multiplier
                            # (-1 * moebius) is structurally identical and just needs sign flip
                            loss = 1.0 - abs(r)
                else:
                    raw_loss = self.huber_loss(y_flat, pred_flat)
                    
                    if sample_weight is not None:
                        loss = np.average(raw_loss, weights=np.asarray(sample_weight).flatten())
                    else:
                        loss = np.mean(raw_loss)

                # 4. Perfect Fit Bypass (No penalty if perfect)
                if loss < self.config.early_stop_mse:
                    tree._cached_mse = loss
                    if cache_key is not None:
                        self._fitness_cache[cache_key] = loss
                    return loss

                # 5. Integer Penalty (The "Race Problem" Fix)
                try:
                    target_is_int = np.all(np.equal(np.mod(y_flat, 1), 0))
                    if target_is_int:
                        pred_is_dirty = not np.all(np.abs(pred_flat - np.round(pred_flat)) < 1e-6)
                        if pred_is_dirty:
                            loss += 10.0
                except Exception:
                    pass
                # 6. Constant Penalty (The "Lazy Constant" Fix)
                if not tree.contains_variables():
                    loss += 100.0

                # 7. Depth penalty (exponential beyond limit)
                tree_depth = tree.depth()
                if tree_depth > getattr(self.config, 'max_tree_depth', 15):
                    excess = tree_depth - getattr(self.config, 'max_tree_depth', 15)
                    loss += 0.1 * (2 ** excess)

                # 8. Repetition penalty (anti-nesting)
                chain_length = tree.max_operator_chain_length()
                if chain_length > 3:
                    loss += 0.5 * (chain_length - 3)

                final_fitness = loss + (self.config.parsimony_coefficient * complexity)
                
                # Cache PENALIZED fitness for Pareto updates
                tree._cached_mse = final_fitness
                if cache_key is not None:
                    self._fitness_cache[cache_key] = final_fitness
                
                return final_fitness

            except Exception:
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
            local_dict.update({'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'max': sp.Max, 'min': sp.Min, 'median': lambda *args: sp.Max(*args)}) # Hack for median until supported
            # Merge with allowed names for comprehensive coverage
            full_local_dict = {**ALLOWED_SYMPY_NAMES, **local_dict}
            
            for seed_str in combined_seeds[:n_individuals // 2]:
                try:
                    # SECURITY: Use AST-based safe parser instead of eval-based sympify
                    expr = safe_sympy_parse(seed_str, local_dict=full_local_dict)
                    tree = ExpressionTree.from_sympy(expr, variables)
                    tree.age = 0
                    population.append(tree)
                except (ValueError, TypeError, SyntaxError, AttributeError) as e:
                    if self.config.verbose: print(f"DEBUG: Seed injection failed for '{seed_str}': {e}")
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
            if tree.fitness is None or getattr(tree, 'age', 0) == 0:
                tree.fitness = self.calculate_fitness(tree, X, y, sample_weight)
            if hasattr(tree, 'age'):
                tree.age += 1
            else:
                tree.age = 1
        # if self.config.verbose: print(f"DEBUG: [Gen {generation}] Evaluation done.")
            
        # 2. Elitism
        population.sort(key=lambda t: t.fitness)
        new_pop = [t.copy() for t in population[:self.config.elitism]]
        
        # ELITE RESCUE: Prevent "Lazy Constant" Problem in Transformed Spaces
        # If ALL elites are constants, inject a variable into one of them.
        # This ensures the gene pool retains variable-containing expressions
        # even when low-variance targets favor constants.
        all_elites_constant = all(not e.contains_variables() for e in new_pop)
        if all_elites_constant and new_pop and new_pop[0].variables:
            from .operators import inject_variable_mutation
            # Mutate the best elite to have a variable
            new_pop[0] = inject_variable_mutation(new_pop[0])
            new_pop[0].fitness = None  # Force re-evaluation
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
                
                # Crossover Rescue: Inject variable if offspring are constant-only
                if not off1.contains_variables() and off1.variables:
                    from .operators import inject_variable_mutation
                    off1 = inject_variable_mutation(off1)
                    off1.fitness = None
                    
                new_pop.append(off1)
                if len(new_pop) < self.config.population_size:
                    if not off2.contains_variables() and off2.variables:
                        from .operators import inject_variable_mutation
                        off2 = inject_variable_mutation(off2)
                        off2.fitness = None
                    new_pop.append(off2)
            else:
                # Mutation
                r = random.random()
                parent = parent1.copy()
                # 40% Point, 25% Insert, 15% Composition, 10% Hoist, 10% Shrink
                if r < 0.40:
                    child = point_mutation(parent, self.config.mutation_rate, self.config.operators)
                elif r < 0.65:  # 25% chance to insert (wrap in unary)
                    from .operators import insert_mutation
                    child = insert_mutation(parent, self.config.operators)
                elif r < 0.80:  # 15% chance to compose (wrap in binary with new branch)
                    from .operators import composition_mutation
                    child = composition_mutation(parent, 3, self.config.operators)
                elif r < 0.90:
                    child = hoist_mutation(parent)
                else:
                    child = shrink_mutation(parent)
                child.age = 0
                # CRITICAL: Reset fitness to prevent zombies
                child.fitness = None
                
                # Genetic Rescue: If mutation produced a constant-only tree,
                # inject a variable to prevent population collapse to constants
                if not child.contains_variables():
                    from .operators import inject_variable_mutation
                    child = inject_variable_mutation(child)
                    child.fitness = None  # Reset fitness after modification
                    
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
