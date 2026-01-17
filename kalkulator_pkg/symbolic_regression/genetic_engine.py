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
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # 4. Boosting Loop
        self.boosted_models = []
        y_residual = y_train.copy()
        
        rounds = self.config.boosting_rounds
        if rounds < 1: rounds = 1
        
        start_time_global = time.time() # For global timeout tracking
        
        # Parallel Config
        n_jobs = getattr(self.config, 'n_jobs', 1)
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        use_parallel = n_jobs > 1 and self.config.n_islands > 1
        
        for round_idx in range(rounds):
            if self.config.verbose and rounds > 1:
                print(f"--- Boosting Round {round_idx + 1}/{rounds} ---")
            
            # Initialize Islands
            islands = self._initialize_islands(variable_names, X_train, y_residual)
            
            # Run Evolution (Generations)
            self._run_evolution_generations(
                islands, X_train, y_residual, X_val, y_val, sample_weight,
                use_parallel, n_jobs, start_time_global
            )
            
            # End of Round: Boosting
            best_round = self.pareto_front.get_best()
            if not best_round: 
                break 
            
            if rounds == 1:
                self.best_tree = best_round.tree
                break
                
            # Update Residuals for Next Round
            success, y_residual = self._update_boosting_residuals(round_idx, X_train, y_residual)
            if not success:
                break
                
            # Clear Pareto Front for next round (fresh search on residuals)
            self.pareto_front = ParetoFront()
            
        # Final model logic
        if hasattr(self, 'boosted_models') and self.boosted_models:
            self.best_tree = self.boosted_models[-1][1]  # Use last tree as representative
        
        # FINAL OPTIMIZATION STEP (Gold Standard)
        # Polish the single best tree (if boosting rounds=1) or the last boosted tree
        if self.best_tree and self.config.constant_optimization_rate > 0:
             try:
                 from .operators import optimize_constants_bfgs
                 if self.config.verbose:
                     print("Running final constant optimization (BFGS)...")
                 
                 # If using boosting with multiple rounds, we optimize the last tree specifically against residuals
                 # If rounds=1, we optimize against original X, y
                 
                 target_for_opt = y_train
                 # If boosting, we should technically optimize against the *residuals* that this tree was trained on
                 # But self.best_tree here is just the last one.
                 # Let's be safe and only aggressively optimize if it's a single-tree model (common case for physics)
                 # Or if we have the residuals handy (y_residual is from loop end).
                 
                 if rounds > 1:
                      target_for_opt = y_residual # Approximates the target for the last stage
                 else:
                      target_for_opt = y_train
                      
                 # Run BFGS
                 # Use a higher max_iter for final polish
                 optimized_tree = optimize_constants_bfgs(
                     self.best_tree, X_train, target_for_opt, 
                     max_iter=100, 
                     sample_weight=sample_weight
                 )
                 
                 # Only accept if better
                 fitness_old = self.strategies['evolution'].calculate_fitness(self.best_tree, X_train, target_for_opt, sample_weight)
                 fitness_new = self.strategies['evolution'].calculate_fitness(optimized_tree, X_train, target_for_opt, sample_weight)
                 
                 if fitness_new <= fitness_old:
                     self.best_tree = optimized_tree
                     # Update Pareto Front with this better version
                     sol = ParetoSolution(
                        expression=optimized_tree.to_pretty_string(),
                        sympy_expr=None,
                        mse=fitness_new,
                        complexity=optimized_tree.complexity(),
                        tree=optimized_tree
                     )
                     self.pareto_front.add(sol)
                     
             except Exception as e:
                 if self.config.verbose:
                     print(f"Final optimization failed: {e}")
            
        return self.pareto_front

    def _split_data(self, X, y):
        """Simple data splitter."""
        if len(y) < 20: return X, X, y, y
        try:
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except ImportError:
            return X, X, y, y

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
        best = self.pareto_front.get_best()
        return best.expression if best else ""

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
        try:
            predictions = best_round.tree.evaluate(X_train)
            if isinstance(predictions, np.ndarray):
                y_new_residual = y_residual - learning_rate * predictions
            else:
                y_new_residual = y_residual - learning_rate * float(predictions)
                
            if self.config.verbose:
                residual_mse = np.mean(y_new_residual ** 2)
                print(f"Round {round_idx + 1}: Residual MSE = {residual_mse:.4e}")
            
            return True, y_new_residual
        except (ValueError, TypeError):
            if self.config.verbose:
                print(f"Round {round_idx + 1}: Failed to update residuals, stopping")
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
