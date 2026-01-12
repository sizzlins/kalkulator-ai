"""Benchmarks Module.

Provides the Feynman Symbolic Regression Benchmark for evaluating
symbolic regression algorithms.

Components:
    - feynman_equations: Database of 100+ physics equations
    - benchmark_runner: Automated benchmarking and comparison
"""

from .benchmark_runner import BenchmarkResult
from .benchmark_runner import BenchmarkSuite
from .benchmark_runner import check_symbolic_equivalence
from .benchmark_runner import quick_benchmark
from .benchmark_runner import run_feynman_benchmark
from .benchmark_runner import run_single_benchmark
from .feynman_equations import FEYNMAN_EQUATIONS
from .feynman_equations import FeynmanEquation
from .feynman_equations import get_equation
from .feynman_equations import get_equations_by_complexity
from .feynman_equations import list_equations

__all__ = [
    # Feynman Equations
    "FeynmanEquation",
    "FEYNMAN_EQUATIONS",
    "get_equation",
    "get_equations_by_complexity",
    "list_equations",
    # Benchmarking
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_single_benchmark",
    "run_feynman_benchmark",
    "quick_benchmark",
    "check_symbolic_equivalence",
]
