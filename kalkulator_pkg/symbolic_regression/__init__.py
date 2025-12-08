"""Symbolic Regression Module.

This module provides genetic programming-based symbolic regression
for discovering mathematical equations from data.

Main Components:
    - ExpressionTree: Tree-based representation of mathematical expressions
    - GeneticSymbolicRegressor: Evolutionary algorithm for equation discovery
    - ParetoFront: Multi-objective optimization results

Example:
    >>> from kalkulator_pkg.symbolic_regression import discover_equation
    >>> import numpy as np
    >>> X = np.linspace(0, 10, 100).reshape(-1, 1)
    >>> y = 3 * X[:, 0]**2 + 2 * X[:, 0] + 1  # Unknown function
    >>> expr, mse, pareto = discover_equation(X, y, variable_names=['x'])
    >>> print(f"Discovered: {expr}")
    >>> print(f"MSE: {mse:.6e}")
"""

from .expression_tree import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
    ExpressionNode,
    ExpressionTree,
    NodeType,
)
from .genetic_engine import GeneticConfig, GeneticSymbolicRegressor, discover_equation
from .operators import (
    apply_mutation,
    constant_optimization,
    crossover,
    hoist_mutation,
    point_mutation,
    shrink_mutation,
    subtree_mutation,
    tournament_selection,
)
from .pareto_front import ParetoFront, ParetoSolution

__all__ = [
    # Expression Trees
    "ExpressionTree",
    "ExpressionNode",
    "NodeType",
    "UNARY_OPERATORS",
    "BINARY_OPERATORS",
    # Genetic Operators
    "point_mutation",
    "subtree_mutation",
    "hoist_mutation",
    "shrink_mutation",
    "constant_optimization",
    "crossover",
    "tournament_selection",
    "apply_mutation",
    # Pareto Optimization
    "ParetoFront",
    "ParetoSolution",
    # Main Algorithm
    "GeneticSymbolicRegressor",
    "GeneticConfig",
    "discover_equation",
]
