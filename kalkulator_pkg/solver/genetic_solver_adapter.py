"""
Genetic Solver Adapter — wraps GeneticSymbolicRegressor for FinderStrategy.

Provides a clean `solve()` interface that converts list-of-tuples data
to numpy arrays, runs the genetic engine, and returns the standard
(success, func_str, factored, error) tuple.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def solve(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False,
    timeout: float = 30.0,
    generations: int = 50,
    population_size: int = 100,
    banned_operators: Optional[set] = None,
    seeds: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Run genetic symbolic regression on the given data.
    """
    import sys
    with open("debug_genetic.log", "a") as f:
        f.write(f"DEBUG: SOLVE CALLED. verbose={verbose}, seeds={seeds}\n")
    try:
        from ..symbolic_regression.genetic_engine import GeneticSymbolicRegressor
        from ..symbolic_regression.genetic_config import GeneticConfig
    except ImportError as e:
        return (False, None, None, f"Genetic engine not available: {e}")

    # --- Convert data to numpy arrays ---
    try:
        X_list = []
        y_list = []
        is_complex = False
        
        for x_tuple, y_val in data_points:
            if isinstance(x_tuple, (list, tuple, np.ndarray)):
                X_list.append([float(v) for v in x_tuple])
            else:
                X_list.append([float(x_tuple)])
            
            # Check for complex values in y
            if isinstance(y_val, complex) or (isinstance(y_val, (int, float)) and False):
                 pass # simplified check
            
            # Use complex check on the raw value
            if isinstance(y_val, complex):
                is_complex = True
            y_list.append(y_val)

        X = np.array(X_list, dtype=np.float64)
        
        # Smart Type Inference for y
        if is_complex or any(isinstance(y, complex) for y in y_list):
             y = np.array(y_list, dtype=np.complex128)
             # Also convert X to complex to allow domain extension (e.g. sqrt(-1) -> 1j)
             X = np.array(X_list, dtype=np.complex128)
             if verbose: print("  [Genetic] Detected Complex Numbers in target y. Converting X to complex.")
        else:
             y = np.array(y_list, dtype=np.float64)

    except (ValueError, TypeError) as e:
        return (False, None, None, f"Cannot convert data to numeric arrays: {e}")

    # --- Filter inf/nan ---
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    n_filtered = int(np.sum(~finite_mask))
    if n_filtered > 0:
        X = X[finite_mask]
        y = y[finite_mask]
        if verbose:
            print(f"  [Genetic] Filtered {n_filtered} non-finite point(s).")

    if len(y) < 3:
        return (False, None, None, "Not enough data points for genetic regression (need 3+).")

    # --- Configure engine ---
    config = GeneticConfig(
        population_size=population_size,
        generations=generations,
        timeout=timeout,
        verbose=verbose,
        n_islands=2,
        boosting_rounds=1,
        early_stop_mse=1e-10,
    )

    # Remove banned operators from config if specified
    if banned_operators:
        original_ops = list(config.operators)
        config.operators = [op for op in original_ops if op not in banned_operators]
        if verbose and len(config.operators) < len(original_ops):
            removed = set(original_ops) - set(config.operators)
            print(f"  [Genetic] Removed banned operators: {removed}")

    # --- Run engine ---
    regressor = GeneticSymbolicRegressor(config=config)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pass seeds down to fit_with_transformations
            best_expr, best_mse, space_name = regressor.fit_with_transformations(
                X, y, param_names, seeds=seeds
            )
    except Exception as e:
        logger.debug(f"Genetic engine error: {e}", exc_info=True)
        return (False, None, None, f"Genetic engine failed: {e}")

    if not best_expr or best_expr.strip() == "":
        return (False, None, None, "Genetic engine found no expression.")

    # --- Post-process: ban enforcement on result string ---
    if banned_operators:
        expr_lower = best_expr.lower()
        for banned in banned_operators:
            if banned.lower() in expr_lower:
                if verbose:
                    print(f"  [Genetic] Result '{best_expr}' contains banned operator '{banned}', rejecting.")
                return (False, None, None, f"Best expression uses banned operator '{banned}'.")

    # --- Quality gate ---
    # Fix: User R2 score instead of absolute MSE > 1.0 to handle unscaled data.
    # If data is y=1000x, MSE might be 100.0 (good), but absolute check rejects it.
    y_variance = np.var(y)
    
    # Handle low variance (nearly constant data)
    # Use ABS for variance in case of complex numbers (var is real, but let's be safe)
    if abs(y_variance) < 1e-12:
        # We expect a very tight fit (MSE near zero)
        if abs(best_mse) > 1e-5:
             if verbose:
                 print(f"  [Genetic] Constant/Low-Var data, but fit invalid (MSE={best_mse:.4e}). Rejecting.")
             return (False, None, None, f"Genetic engine result has poor fit (MSE={best_mse:.4e}).")
    else:
        # Normal data: Check R2
        # R2 = 1 - (SS_res / SS_tot) = 1 - (MSE / Var)
        # Use abs() for robust complex comparison
        r2 = 1.0 - (abs(best_mse) / abs(y_variance))
        
        # Require at least 1% variance explained for a non-trivial model
        if r2 < 0.01: 
            if verbose:
                print(f"  [Genetic] Best fit explains <1% variance (R2={r2:.4f}, MSE={best_mse:.4e}). Rejecting.")
            return (False, None, None, f"Genetic engine result has poor fit (R2={r2:.4f}).")

    if verbose:
        print(f"  [Genetic] Found: {best_expr} (MSE={best_mse:.4e}, space={space_name})")

    return (True, best_expr, None, None)
