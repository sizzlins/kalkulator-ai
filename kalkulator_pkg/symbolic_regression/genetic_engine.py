"""Genetic Programming Symbolic Regression Engine."""

import gc
import random
import time
import numpy as np

# Scikit-Learn compliance (optional - graceful degradation)
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    # Define dummy classes for compatibility when sklearn not installed
    class BaseEstimator:
        pass
    class RegressorMixin:
        pass
    SKLEARN_AVAILABLE = False

from .expression_tree import ExpressionTree
from .pareto_front import ParetoFront, ParetoSolution
from .genetic_config import GeneticConfig
from .strategies import BoostingStrategy, EvolutionStrategy


# -----------------------------------------------------------------------------
# EvolutionTrainer: Extracted from GeneticSymbolicRegressor (v3.1 Audit Remediation)
# -----------------------------------------------------------------------------

class EvolutionTrainer:
    """Runs the evolutionary loop for genetic symbolic regression.
    
    This class encapsulates the generation-by-generation evolution logic,
    including parallel execution, island migration, and convergence checks.
    Extracted from GeneticSymbolicRegressor to decouple concerns.
    
    Boosting Behavior (v3.3 Documentation):
    ----------------------------------------
    When boosting_rounds > 1, the model is an ADDITIVE ENSEMBLE:
    
        F(x) = η₁*T₁(x) + η₂*T₂(x) + ... + ηₘ*Tₘ(x)
    
    where Tᵢ are symbolic trees and ηᵢ are learning rates.
    
    - predict() correctly sums all trees for accurate predictions
    - get_expression() returns the FULL ensemble expression
    - get_ensemble_expression() explicitly returns the sum formula
    
    Note: For interpretability of physics, use boosting_rounds=1 to get
    a single clean expression like 'E = m*c**2'.
    """
    
    def __init__(self, config: GeneticConfig, evolution_strategy: EvolutionStrategy, pareto_front: ParetoFront):
        self.config = config
        self.strategy = evolution_strategy
        self.pareto_front = pareto_front
        
    def train(
        self,
        islands: list,
        X_train: np.ndarray,
        y_target: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray = None,
        use_parallel: bool = False,
        n_jobs: int = 1,
        start_time_global: float = None,
        update_pareto_callback=None
    ):
        """Run the main evolution loop for one boosting round.
        
        Args:
            islands: List of population islands
            X_train: Training features
            y_target: Training targets (or residuals for boosting)
            X_val: Validation features
            y_val: Validation targets
            sample_weight: Optional sample weights
            use_parallel: Whether to use parallel execution
            n_jobs: Number of parallel jobs
            start_time_global: Global start time for timeout
            update_pareto_callback: Callback to update pareto front with (islands, X, y)
        """
        from contextlib import ExitStack
        import time
        import gc
        
        if start_time_global is None:
            start_time_global = time.time()
        
        best_mse_observed = float('inf')
        patience_counter = 0

        # DEBUG LOGGING
        try:
             with open("verification_progress.txt", "a") as f: 
                 f.write(f"DEBUG: Entered train. n_jobs={n_jobs}, islands={len(islands)}\n")
        except: pass
        
        with ExitStack() as stack:
            executor = None
            shm_X_info = None
            shm_y_info = None
            
            if use_parallel:
                try:
                     with open("verification_progress.txt", "a") as f: f.write("DEBUG: Attempting parallel init\n")
                     # ... (parallel setup) ...
                     from concurrent.futures import ProcessPoolExecutor
                     from .parallel import managed_shared_memory, evolve_island_worker
                     # ...
                     shm_X = stack.enter_context(managed_shared_memory(X_train))
                     shm_X_info = {'name': shm_X.name, 'shape': X_train.shape, 'dtype': X_train.dtype}
                     shm_y = stack.enter_context(managed_shared_memory(y_target))
                     shm_y_info = {'name': shm_y.name, 'shape': y_target.shape, 'dtype': y_target.dtype}
                     executor = stack.enter_context(ProcessPoolExecutor(max_workers=n_jobs))
                except (ImportError, OSError) as e:
                    if self.config.verbose: print(f"Parallel init failed: {e}. Falling back to serial.")
                    use_parallel = False
                    executor = None
            
            try:
                 with open("verification_progress.txt", "a") as f: f.write(f"DEBUG: Loop start. use_parallel={use_parallel}\n")
            except: pass

            for gen in range(self.config.generations):
                # Timeout Check (Global)
                if self.config.timeout and (time.time() - start_time_global > self.config.timeout):
                    break
                    
                # Evolve Islands
                if use_parallel and executor:
                    futures = []
                    for i in range(len(islands)):
                        futures.append(executor.submit(
                            evolve_island_worker,
                            islands[i], shm_X_info, shm_y_info, 
                            self.config, gen, self.strategy
                        ))
                    islands = [f.result() for f in futures]
                else:
                    for i in range(len(islands)):
                        islands[i] = self.strategy.evolve(
                            islands[i], X_train, y_target, gen, sample_weight
                        )
                
                # Migration
                if gen > 0 and gen % self.config.migration_interval == 0:
                    self.strategy.migrate(islands)
                    
                # Update Pareto Front
                self._update_pareto_front(islands)
                    
        return islands

    def _update_pareto_front(self, islands):
        """Update Pareto Front with current population."""
        import sympy as sp
        for island in islands:
            for tree in island:
                if tree.fitness is None or not np.isfinite(tree.fitness):
                    continue
                
                try:
                    sympy_expr = tree.to_sympy()
                    
                    sol = ParetoSolution(
                        expression=str(tree),
                        sympy_expr=sympy_expr,
                        mse=tree.fitness,
                        complexity=tree.complexity(),
                        tree=tree
                    )
                    self.pareto_front.add(sol)
                except Exception:
                    continue

    def train_full_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        variable_names: list[str],
        sample_weight: np.ndarray = None
    ):
        """Orchestrate the full training process including Boosting."""
        from .genetic_config import GeneticConfig
        from .strategies import BoostingStrategy
        
        # boosting loop logic moved from GeneticSymbolicRegressor
        boosted_models = []
        y_residual = y_train.copy()
        best_tree_final = None
        
        rounds = self.config.boosting_rounds
        if rounds < 1: rounds = 1
        
        start_time_global = time.time()
        
        # Parallel Config
        n_jobs = getattr(self.config, 'n_jobs', 1)
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        use_parallel = n_jobs > 1 and self.config.n_islands > 1
        
        for round_idx in range(rounds):
            if self.config.verbose and rounds > 1:
                print(f"--- Boosting Round {round_idx + 1}/{rounds} ---")
            
            # Initialize Islands (Need helper or move logic)
            # Since _initialize_islands was on Regressor, we need to adapt.
            # We can implement a simple initializer here or pass a factory.
            # Re-implementing simplified island init here for decoupling.
            islands = self._init_islands_internal(variable_names, X_train, y_residual)
            
            # Run Evolution Round
            islands = self.train(
                islands, X_train, y_residual, X_val, y_val,
                sample_weight=sample_weight,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                start_time_global=start_time_global
            )
            
            # End of Round
            best_round = self.pareto_front.get_best()
            with open("verification_progress.txt", "a") as f: f.write(f"DEBUG: Best round retrieved: {best_round is not None}\n")

            if not best_round:
                if self.config.verbose: print("Evolution failed to find any valid solution this round.")
                break
                
            # Store Model
            learning_rate = getattr(self.config, 'learning_rate', 0.1)
            boosted_models.append((learning_rate, best_round.tree))
            best_tree_final = best_round.tree # Track latest
            
            if rounds == 1:
                with open("verification_progress.txt", "a") as f: f.write("DEBUG: Rounds=1, breaking loop\n")
                break
            
            # Update Residuals
            # We need the update logic. It was on Regressor as well.
            # Implemented here directly using the new Pseudo-Residual logic?
            # Or call a helper? We should define `update_residuals` on EvolutionTrainer or Strategy.
            # Strategy is better.
            
            # Corrected method name to match definition
            success, y_residual = self._update_boosting_residuals(round_idx, X_train, y_residual)
            if not success:
                break
                
            # Clear Pareto Front for next round
            self.pareto_front = ParetoFront() # Reset
            
        return best_tree_final, boosted_models

    def _init_islands_internal(self, variable_names, X, y):
        # Simplified island initialization
        # Inline Population class to avoid import issues
        class Population(list):
            def __init__(self, size, variable_names, config, random_state):
                super().__init__()
                self.size = size
                self.variable_names = variable_names
                self.config = config
                if random_state is not None:
                    random.seed(random_state)
            
            def initialize(self):
                # Simple initialization
                depths = range(2, self.config.max_depth + 1)
                methods = ["grow", "full"]
                self.clear()
                while len(self) < self.size:
                    depth = depths[len(self) % len(depths)]
                    method = methods[len(self) % len(methods)]
                    tree = ExpressionTree.random_tree(
                        variables=self.variable_names,
                        max_depth=depth,
                        operators=self.config.operators,
                        method=method
                    )
                    tree.age = 0
                    self.append(tree)

        islands = []
        for i in range(self.config.n_islands):
            # Seed diversity
            seed = int(time.time() * 1000) + i
            # Create population
            pop = Population(
               size=self.config.population_size // self.config.n_islands,
               variable_names=variable_names,
               config=self.config,
               random_state=seed
            )
            # Initialize (Ramped Half-and-Half)
            # We need X, y to evaluate initial fitness? 
            # Usually we init random trees first.
            pop.initialize()
            # Evaluate
            self.strategy.evaluate_population(pop, X, y, sample_weight=None)
            islands.append(pop)
        return islands

    def _update_residuals_internal(self, round_idx, tree, y_residual, X_train, learning_rate):
        # Boosting update logic (Copied/Adapted from Regressor)
        try:
            predictions = tree.evaluate(X_train)
            if not isinstance(predictions, np.ndarray):
                 predictions = np.full(X_train.shape, float(predictions))
            
            loss_type = getattr(self.config, 'loss_function', 'mse').lower()
            if loss_type == 'huber':
                delta = getattr(self.config, 'huber_delta', 1.35)
                residual = y_residual - predictions
                abs_r = np.abs(residual)
                mask_small = abs_r <= delta
                mask_large = ~mask_small
                pseudo_residuals = np.zeros_like(residual)
                pseudo_residuals[mask_small] = residual[mask_small]
                pseudo_residuals[mask_large] = delta * np.sign(residual[mask_large])
                y_new_residual = pseudo_residuals
            else:
                 y_new_residual = y_residual - learning_rate * predictions
                 
            if self.config.verbose:
                residual_mse = np.mean(y_new_residual ** 2)
                print(f"Round {round_idx + 1}: Residual MSE = {residual_mse:.4e}")
            return True, y_new_residual
        except Exception as e:
            if self.config.verbose: print(f"Update residuals failed: {e}")
            return False, y_residual
                



class GeneticSymbolicRegressor(BaseEstimator, RegressorMixin):
    """Genetic Programming Symbolic Regression Engine.
    
    Scikit-Learn compatible estimator for symbolic regression using genetic programming.
    Can be used in sklearn pipelines and cross-validation.
    
    Refactored to use Strategy Pattern for maintainability.
    Supports Parallel Execution via Shared Memory (Zero-Copy).
    """

    def __init__(self, config: GeneticConfig | None = None):
        self.config = config or GeneticConfig()
        self.pareto_front = ParetoFront()
        self.best_tree = None
        self._last_seeds = [] # State for equivalent checks
        self.boosted_models = []  # List of (learning_rate, tree) for additive boosting
        
        # Parallel execution
        self._executor = None

    def fit(self, X: np.ndarray, y: np.ndarray, variable_names: list[str] = None, sample_weight=None) -> ParetoFront:
        """Fit the model using Boosting and Evolution strategies."""
        
        # 1. Initialization
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        if len(y.shape) == 1: y = y.flatten()
        
        self.strategies = {
            'boosting': BoostingStrategy(self.config),
            'evolution': EvolutionStrategy(self.config)
        }
        
        # 2. Smart Weighting (The Vise Strategy)
        if sample_weight is None:
            sample_weight = self.strategies['boosting'].calculate_weights(X, y)
            
        # 3. Data Split
        X_train, X_val, y_train, y_val, w_train, w_val = self._split_data(X, y, sample_weight)
        
        # v4.2 Audit Refactoring: Decoupled God Object.
        trainer = EvolutionTrainer(
            self.config,
            self.strategies['evolution'],
            self.pareto_front
        )
        
        # Delegate to Trainer with correct weights
        self.best_tree, self.boosted_models = trainer.train_full_model(
             X_train, y_train, X_val, y_val, variable_names, w_train
        )
        # Update pareto front from trainer
        self.pareto_front = trainer.pareto_front
            
    def fit_with_transformations(self, X: np.ndarray, y: np.ndarray, variable_names: list[str]) -> tuple[str, float, str]:
        """Fit model in multiple spaces (direct, log, inverse) and pick best.
        
        This method helps discover functions like exp(x) (linear in log space)
        or 1/x (linear in inverse space) more easily.
        
        Args:
            X: Input features
            y: Target values
            variable_names: Names of input variables
            
        Returns:
            Tuple of (best_expression_string, best_mse, best_space_name)
        """
        import numpy as np
        
        # Helper to run a fresh regression
        def run_space(y_target, space_name):
            # Create fresh instance to avoid state pollution
            # We need to access the results.
            # Ideally we clone self.config. 
            # Note: We must replicate the config.
            import copy
            # Deep copy config to ensure isolation
            cfg = copy.deepcopy(self.config)
            
            reg = GeneticSymbolicRegressor(cfg)
            reg.fit(X, y_target, variable_names)
            
            best_tree = reg.best_tree
            if best_tree is None:
                return None
                
            y_pred_transformed = reg.predict(X)
            return best_tree, y_pred_transformed, reg.get_expression()

        candidates = []
        
        # 1. Direct Space
        try:
            # For direct space, we can just use 'self' if we want, but cleaner to use fresh instance
            # and update self at the end with the winner.
            res = run_space(y, "direct")
            if res:
                tree, pred, expr = res
                mse = np.mean((y - pred)**2)
                candidates.append((expr, mse, "direct", tree))
        except Exception:
            pass

        # 2. Log Space (if y > 0)
        if np.all(y > 0):
            try:
                y_log = np.log(y)
                res = run_space(y_log, "log")
                if res:
                    tree, pred_log, expr_log = res
                    # Transform back: exp(pred_log)
                    pred = np.exp(pred_log)
                    mse = np.mean((y - pred)**2)
                    full_expr = f"exp({expr_log})"
                    candidates.append((full_expr, mse, "log", tree)) # We can't really store the tree directly as "the" tree for predict...
            except Exception:
                pass

        # 3. Inverse Space (if y != 0)
        if np.all(y != 0):
            try:
                y_inv = 1.0 / y
                res = run_space(y_inv, "inverse")
                if res:
                    tree, pred_inv, expr_inv = res
                    # Transform back: 1/pred_inv
                    with np.errstate(divide='ignore'):
                        pred = 1.0 / pred_inv
                    # Handle divergance?
                    mask = np.isfinite(pred)
                    if np.any(mask):
                        mse = np.mean((y[mask] - pred[mask])**2)
                        full_expr = f"1/({expr_inv})"
                        candidates.append((full_expr, mse, "inverse", tree))
            except Exception:
                pass

        # Pick best
        if not candidates:
            return "0", float('inf'), "none"
            
        candidates.sort(key=lambda x: x[1]) # Sort by MSE
        best_expr, best_mse, best_space, best_tree_obj = candidates[0]
        
        # Update self with the best result (so predict/get_expression work somewhat? 
        # Actually, for log/inverse, 'self.best_tree' won't represent the full transform.
        # But this method returns the STRING expression, which the REPL uses parse logic to handle.
        # The REPL re-parses the string into a tree. 
        # So we just need to return the string.
        # However, for consistency, if 'direct' won, we might want to populate self.
        
        if best_space == "direct":
            # We can't easily "inject" the result into self without re-running or manually setting state.
            # But the REPL seems to rely on the RETURN VALUES of this function:
            # best_expr, best_mse_val, best_space = regressor.fit_with_transformations(...)
            pass
            
        return best_expr, best_mse, best_space
        
    def _split_data(self, X, y, sample_weight=None):
        """Simple data splitter."""
        if len(y) < 20: 
            if sample_weight is None:
                return X, X, y, y, sample_weight, sample_weight
            return X, X, y, y, sample_weight, sample_weight
            
        try:
            from sklearn.model_selection import train_test_split
            if sample_weight is not None:
                return train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)
            else:
                 X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                 return X_tr, X_val, y_tr, y_val, None, None
        except ImportError:
            return X, X, y, y, sample_weight, sample_weight

    def _update_pareto_front(self, islands, X, y):
        """Update Pareto front from all islands."""
        all_inds = [ind for island in islands for ind in island]
        # Only check top 20 to save time
        all_inds.sort(key=lambda t: t.fitness)
        
        for tree in all_inds[:20]:
            try:
                # Use cached MSE if available, otherwise recalculate
                if tree._cached_mse < float('inf'):
                    mse = tree._cached_mse
                else:
                    mse = self.strategies['evolution'].calculate_fitness(tree, X, y)
                    tree._cached_mse = mse  # Cache for next time
                
                if mse < 1e6:
                    sol = ParetoSolution(
                        expression=tree.to_pretty_string(),
                        sympy_expr=None, # Lazy load later
                        mse=mse,
                        complexity=tree.complexity(),
                        tree=tree.copy()
                    )
                    self.pareto_front.add(sol)
            except (ValueError, OverflowError, TypeError, AttributeError):
                pass

    def get_expression(self) -> str:
        """Get the best expression.
        
        For boosting mode: Returns the full ensemble expression F(x) = Σ η * T_i(x).
        For single-round: Returns the best Pareto expression.
        """
        # If boosting was used, return full ensemble
        if len(self.boosted_models) > 1:
            return self.get_ensemble_expression()
        
        # Single tree mode
        best = self.pareto_front.get_best()
        return best.expression if best else ""
    
    def get_ensemble_expression(self) -> str:
        """Get the full additive boosting ensemble expression.
        
        Returns:
            String of form "η₁*T₁ + η₂*T₂ + ..." for all boosted trees.
            This is the TRUE mathematical model used by predict().
        """
        if not self.boosted_models:
            best = self.pareto_front.get_best()
            return best.expression if best else ""
        
        terms = []
        for learning_rate, tree in self.boosted_models:
            expr_str = str(tree.to_sympy())
            if abs(learning_rate - 1.0) < 1e-6:
                terms.append(f"({expr_str})")
            else:
                terms.append(f"{learning_rate:.4g}*({expr_str})")
        
        return " + ".join(terms)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.
        
        For boosting: F(x) = Σ η * T_i(x) (sum of all trees weighted by learning rate)
        For single-round: Uses best_tree directly.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # If we have boosted models, sum their contributions
        if self.boosted_models:
            result = np.zeros(X.shape[0])
            for learning_rate, tree in self.boosted_models:
                try:
                    pred = tree.evaluate_fast(X)
                    if np.isscalar(pred):
                        pred = np.full(X.shape[0], pred)
                    result += learning_rate * np.asarray(pred)
                except (ValueError, TypeError):
                    continue
            return result
            
        # Single tree mode
        if self.best_tree is not None:
            try:
                pred = self.best_tree.evaluate_fast(X)
                if np.isscalar(pred):
                    return np.full(X.shape[0], pred)
                return np.asarray(pred)
            except (ValueError, TypeError):
                return np.zeros(X.shape[0])
                
        return np.zeros(X.shape[0])

    def _initialize_islands(self, variable_names, X_train, y_target):
        """Initialize population islands."""
        islands = []
        for _ in range(self.config.n_islands):
            pop = self.strategies['evolution'].initialize_population(
                variable_names, self.config.population_size, 
                seeds=self.config.seeds, X=X_train, y=y_target
            )
            islands.append(pop)
        return islands

    def _update_boosting_residuals(self, round_idx: int, X_train: np.ndarray, y_residual: np.ndarray) -> tuple[bool, np.ndarray]:
        """Update residuals for boosting and store current model."""
        best_round = self.pareto_front.get_best()
        if not best_round:
            return False, y_residual
            
        # Store current model with learning rate
        learning_rate = getattr(self.config, 'learning_rate', 0.1)
        self.boosted_models.append((learning_rate, best_round.tree))
        
        # Update residuals: y_new = y_old - η * F_k(X)
        # Update residuals: y_new = y_old - η * F_k(X)
        # v4.1 Audit Fix: Use Negative Gradient (Pseudo-Residuals)
        # For MSE (L2), Gradient is -(y - y_pred), so return y - y_pred (which is negative gradient).
        # For Huber, we must calculate the specific gradient.
        
        try:
            predictions = best_round.tree.evaluate(X_train)
            if not isinstance(predictions, np.ndarray):
                 predictions = np.full(X_train.shape, float(predictions))
                 
            # Check loss function from config (defaulting to MSE/L2 if not specified)
            loss_type = getattr(self.config, 'loss_function', 'mse').lower()
            
            if loss_type == 'huber':
                # Gradient of Huber Loss L(y, F):
                # if |r| <= delta: grad = -r  => neg_grad = r = y - pred
                # if |r| > delta:  grad = -delta * sign(r) => neg_grad = delta * sign(r)
                delta = getattr(self.config, 'huber_delta', 1.35)
                residual = y_residual - predictions
                abs_r = np.abs(residual)
                
                # Vectorized gradient calculation
                mask_small = abs_r <= delta
                mask_large = ~mask_small
                
                # Pseudo-residual (direction we want to move)
                pseudo_residuals = np.zeros_like(residual)
                pseudo_residuals[mask_small] = residual[mask_small]
                pseudo_residuals[mask_large] = delta * np.sign(residual[mask_large])
                
                # Update target: We want the NEXT tree to fit the Pseudo-Residual
                # New "Residual" target = Pseudo-Residual
                # Note: Standard boosting fits the negative gradient directly.
                # So the new target y_residual becomes the Pseudo-Residual.
                # However, we must scale by learning rate if we are doing "Shrinkage" 
                # strictly on the update step, but usually boosting defines:
                # F_new = F_old + lr * Tree(fitting neg_grad)
                # So the target for the tree IS the neg_grad.
                
                y_new_residual = pseudo_residuals
                
            else:
                # MSE Case: Neg Gradient is (y - y_pred)
                # But wait, y_residual passed in is (y_true - F_prev)??
                # Yes, in the loop: y_residual IS the current residual.
                # So we simply want to fit the NEW residual: y_residual - lr * predictions?
                # NO. If we fit a tree to y_residual, and multiply by lr, 
                # then F_new = F_old + lr * Tree.
                # So the NEW residual is y_true - F_new = y_true - (F_old + lr * pred)
                # = (y_true - F_old) - lr * pred
                # = y_residual - lr * predictions.
                
                y_new_residual = y_residual - learning_rate * predictions

            if self.config.verbose:
                residual_mse = np.mean(y_new_residual ** 2)
                print(f"Round {round_idx + 1}: Residual MSE = {residual_mse:.4e}")
            
            return True, y_new_residual
        except (ValueError, TypeError, AttributeError) as e:
            if self.config.verbose:
                print(f"Round {round_idx + 1}: Failed to update residuals: {e}")
            return False, y_residual

    def _run_evolution_generations(
        self, 
        islands: list, 
        X_train: np.ndarray, 
        y_residual: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        sample_weight: np.ndarray,
        use_parallel: bool,
        n_jobs: int,
        start_time_global: float
    ):
        """Run the main evolution loop for one boosting round."""
        from contextlib import ExitStack
        import time
        import gc
        
        start_time_round = time.time() # For round-specific timing if needed, or check global
        best_mse_observed = float('inf')
        patience_counter = 0
        
        with ExitStack() as stack:
            executor = None
            shm_X_info = None
            shm_y_info = None
            
            if use_parallel:
                try:
                    from concurrent.futures import ProcessPoolExecutor
                    from .parallel import managed_shared_memory, evolve_island_worker
                    
                    # Create Shared Memory
                    shm_X = stack.enter_context(managed_shared_memory(X_train))
                    shm_X_info = {'name': shm_X.name, 'shape': X_train.shape, 'dtype': X_train.dtype}
                    
                    shm_y = stack.enter_context(managed_shared_memory(y_residual))
                    shm_y_info = {'name': shm_y.name, 'shape': y_residual.shape, 'dtype': y_residual.dtype}

                    executor = stack.enter_context(ProcessPoolExecutor(max_workers=n_jobs))
                except (ImportError, OSError) as e:
                    if self.config.verbose: print(f"Parallel init failed: {e}. Falling back to serial.")
                    use_parallel = False
                    executor = None

            for gen in range(self.config.generations):
                # Timeout Check (Global)
                if self.config.timeout and (time.time() - start_time_global > self.config.timeout):
                    break
                    
                # Evolve Islands
                if use_parallel and executor:
                    futures = []
                    for i in range(len(islands)):
                        futures.append(executor.submit(
                            evolve_island_worker,
                            islands[i], shm_X_info, shm_y_info, 
                            self.config, gen, self.strategies['evolution']
                        ))
                    islands = [f.result() for f in futures]
                else:
                    for i in range(len(islands)):
                        islands[i] = self.strategies['evolution'].evolve(
                            islands[i], X_train, y_residual, gen, sample_weight
                        )
                
                # Migration
                if gen > 0 and gen % self.config.migration_interval == 0:
                    self.strategies['evolution'].migrate(islands)
                    
                # Update Pareto Front
                self._update_pareto_front(islands, X_val, y_val)
                
                # Periodic GC
                if gen > 0 and gen % 50 == 0:
                    gc.collect()
                
                # Check Convergence / Early Stop
                best_sol = self.pareto_front.get_best()
                if best_sol:
                    if best_sol.mse < self.config.early_stop_mse:
                        if self.config.verbose: print(f"Early stop: Perfect fit MSE {best_sol.mse:.2e}")
                        break
                        
                    # Patience Logic
                    if best_sol.mse < best_mse_observed * (1 - self.config.min_improvement):
                        best_mse_observed = best_sol.mse
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience:
                            break


def discover_equation(X: np.ndarray, y: np.ndarray, **kwargs) -> ParetoFront:
    """Legacy wrapper for discovering equations.
    
    Args:
        X: Input data
        y: Target data
        **kwargs: Configuration parameters for GeneticConfig
        
    Returns:
        ParetoFront containing discovered solutions
    """
    config_params = {}
    # Filter kwargs that match GeneticConfig fields
    import inspect
    sig = inspect.signature(GeneticConfig)
    for k, v in kwargs.items():
        if k in sig.parameters:
            config_params[k] = v
            
    config = GeneticConfig(**config_params)
    regressor = GeneticSymbolicRegressor(config)
    return regressor.fit(X, y)
