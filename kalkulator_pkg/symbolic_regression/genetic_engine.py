"""Genetic Programming Symbolic Regression Engine."""

import gc
import random
import time
import numpy as np
import sympy as sp

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
    """

    def __init__(self, config: GeneticConfig | None = None):
        self.config = config or GeneticConfig()
        self.pareto_front = ParetoFront()
        self.best_tree = None
        self._last_seeds = [] # State for equivalent checks
        self.boosted_models = []  # List of (learning_rate, tree) for additive boosting

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
        current_model_tree = None
        y_residual = y_train.copy()
        
        rounds = self.config.boosting_rounds
        if rounds < 1: rounds = 1
        
        for round_idx in range(rounds):
            if self.config.verbose and rounds > 1:
                print(f"--- Boosting Round {round_idx + 1}/{rounds} ---")
                
            # Initialize Islands
            islands = []
            for _ in range(self.config.n_islands):
                pop = self.strategies['evolution'].initialize_population(
                    variable_names, self.config.population_size, 
                    seeds=self.config.seeds, X=X_train, y=y_residual
                )
                islands.append(pop)
                
            # Evolution Loop
            start_time = time.time()
            best_mse_observed = float('inf')
            patience_counter = 0
            
            for gen in range(self.config.generations):
                # Timeout Check
                if self.config.timeout and (time.time() - start_time > self.config.timeout):
                    break
                    
                # Evolve Islands
                for i in range(len(islands)):
                    islands[i] = self.strategies['evolution'].evolve(
                        islands[i], X_train, y_residual, gen, sample_weight
                    )
                    
                # Migration
                if gen > 0 and gen % self.config.migration_interval == 0:
                    self.strategies['evolution'].migrate(islands)
                    
                # Update Pareto Front
                self._update_pareto_front(islands, X_val, y_val)
                
                # Periodic GC to prevent memory fragmentation (every 50 generations)
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
                            
            # End of Round: Update Residuals for Boosting
            best_round = self.pareto_front.get_best()
            if not best_round: break
            
            # (Simplification: If boosting is disabled, we are done)
            if rounds == 1:
                self.best_tree = best_round.tree
                break
                
            # ====== BOOSTING MODEL SUMMATION ======
            # Store current model with learning rate
            learning_rate = getattr(self.config, 'learning_rate', 0.1)
            self.boosted_models.append((learning_rate, best_round.tree))
            
            # Update residuals: y_new = y_old - η * F_k(X)
            try:
                predictions = best_round.tree.evaluate(X_train)
                if isinstance(predictions, np.ndarray):
                    y_residual = y_residual - learning_rate * predictions
                else:
                    y_residual = y_residual - learning_rate * float(predictions)
                    
                if self.config.verbose:
                    residual_mse = np.mean(y_residual ** 2)
                    print(f"Round {round_idx + 1}: Residual MSE = {residual_mse:.4e}")
            except (ValueError, TypeError):
                if self.config.verbose:
                    print(f"Round {round_idx + 1}: Failed to update residuals, stopping")
                break
                
            # Clear Pareto front for next round (fresh search on residuals)
            self.pareto_front = ParetoFront()
            
        # Final model is the sum of all boosted trees
        if hasattr(self, 'boosted_models') and self.boosted_models:
            self.best_tree = self.boosted_models[-1][1]  # Use last tree as representative
            
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
