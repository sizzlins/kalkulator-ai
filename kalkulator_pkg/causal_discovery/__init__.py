"""Causal Discovery Module.

Provides algorithms for discovering causal structure from observational data.

Components:
    - pc_algorithm: Peter-Clark algorithm for causal graph discovery
"""

from .pc_algorithm import (
    CausalGraph,
    PCAlgorithm,
    discover_causal_graph,
    g2_test,
    partial_correlation,
    partial_correlation_test,
)

__all__ = [
    "CausalGraph",
    "PCAlgorithm",
    "partial_correlation",
    "partial_correlation_test",
    "g2_test",
    "discover_causal_graph",
]
