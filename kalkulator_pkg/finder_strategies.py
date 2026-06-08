"""
Function finding strategies using the Strategy Pattern.

Each strategy defines a specific approach for discovering functions from data:
- UnivariateStrategy: Single-variable functions (heuristics -> regression -> genetic)
- MultivariateStrategy: Multi-variable functions (regression -> genetic)
- HybridStrategy: Mixed numeric/symbolic data
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import logging
from .core import Context
from .solver import regression_solver
from .solver import genetic_solver_adapter
from .finder_heuristics import check_triangle_wave, check_advanced_heuristics

logger = logging.getLogger(__name__)


class FinderStrategy(ABC):
    """Abstract base class for function finding strategies."""

    @abstractmethod
    def solve(
        self,
        context: Context,
        numeric_data: List[Tuple[Any, Any]],
        symbolic_data: List[Tuple[Any, Any]],
        param_names: List[str],
        config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        pass


class UnivariateStrategy(FinderStrategy):
    """Strategy for single-variable functions (e.g., f(x)).

    Pipeline:
        1. Triangle Wave heuristic (fast, exact match)
        2. Linear/Polynomial regression (clean output for simple functions)
        3. Advanced heuristics (Rational SVD, Harmonic, etc.)
        4. Genetic engine fallback (AI-powered discovery)
    """

    def solve(
        self,
        context: Context,
        numeric_data: List[Tuple[Any, Any]],
        symbolic_data: List[Tuple[Any, Any]],
        param_names: List[str],
        config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:

        verbose = config.get("verbose", False)
        all_data = numeric_data + symbolic_data

        if not all_data:
            return (False, None, None, "No data provided")

        # 1. Triangle Wave heuristic
        tri_func = check_triangle_wave(all_data, param_names, verbose)
        if tri_func:
            return (True, tri_func, None, "Triangle Wave Heuristic")

        # 2. Linear/Polynomial regression
        res = regression_solver.solve(
            all_data, param_names, verbose=verbose, skip_linear=False
        )
        if res[0]:
            return res

        # 3. Advanced heuristics (Rational SVD, Harmonic, etc.)
        heuristic_func = check_advanced_heuristics(all_data, param_names, verbose)
        if heuristic_func:
            return (True, heuristic_func, None, "Heuristic match")

        # 4. Genetic engine fallback
        if config.get("use_genetic", True):
            if verbose:
                print("  [Strategy] Regression/heuristics failed, trying genetic engine...")
            banned = config.get("banned_operators", None)
            res = genetic_solver_adapter.solve(
                all_data,
                param_names,
                verbose=verbose,
                banned_operators=banned,
            )
            if res[0]:
                return res

        return (False, None, None, "No univariate function found.")


class MultivariateStrategy(FinderStrategy):
    """Strategy for multi-variable functions (e.g., f(x, y)).

    Pipeline:
        1. Linear/Polynomial regression
        2. Genetic engine fallback (AI-powered discovery)
    """

    def solve(
        self,
        context: Context,
        numeric_data: List[Tuple[Any, Any]],
        symbolic_data: List[Tuple[Any, Any]],
        param_names: List[str],
        config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:

        verbose = config.get("verbose", False)
        all_data = numeric_data + symbolic_data

        if not all_data:
            return (False, None, None, "No data provided")

        # 1. Linear/Polynomial regression
        res = regression_solver.solve(
            all_data, param_names, verbose=verbose, skip_linear=False
        )
        if res[0]:
            return res

        # 2. Genetic engine fallback
        if config.get("use_genetic", True):
            if verbose:
                print("  [Strategy] Regression failed, trying genetic engine...")
            banned = config.get("banned_operators", None)
            res = genetic_solver_adapter.solve(
                all_data,
                param_names,
                verbose=verbose,
                banned_operators=banned,
            )
            if res[0]:
                return res

        return (False, None, None, "No multivariate function found.")


class HybridStrategy(FinderStrategy):
    """Strategy for mixed numeric/symbolic data or complex constraints."""

    def solve(
        self,
        context: Context,
        numeric_data: List[Tuple[Any, Any]],
        symbolic_data: List[Tuple[Any, Any]],
        param_names: List[str],
        config: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:

        if len(param_names) == 1:
            return UnivariateStrategy().solve(
                context, numeric_data, symbolic_data, param_names, config
            )
        else:
            return MultivariateStrategy().solve(
                context, numeric_data, symbolic_data, param_names, config
            )
