"""Benchmarks Module.

Provides the Feynman Symbolic Regression Benchmark for evaluating
symbolic regression algorithms.

Components:
    - feynman_equations: Database of 100+ physics equations
    - benchmark_runner: Automated benchmarking and comparison
"""

from .feynman_equations import (
    FeynmanEquation,
    FEYNMAN_EQUATIONS,
    get_equation,
    get_equations_by_complexity,
    list_equations,
)
from .benchmark_runner import (
    BenchmarkResult,
    BenchmarkSuite,
    run_single_benchmark,
    run_feynman_benchmark,
    quick_benchmark,
    check_symbolic_equivalence,
)

__all__ = [
    # Feynman Equations
    'FeynmanEquation',
    'FEYNMAN_EQUATIONS',
    'get_equation',
    'get_equations_by_complexity',
    'list_equations',
    
    # Benchmarking
    'BenchmarkResult',
    'BenchmarkSuite',
    'run_single_benchmark',
    'run_feynman_benchmark',
    'quick_benchmark',
    'check_symbolic_equivalence',
]
