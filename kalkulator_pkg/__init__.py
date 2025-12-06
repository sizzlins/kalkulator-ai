"""Kalkulator package: modularized components for parser, solver, worker, and CLI."""

__version__ = "1.0.0"

__all__ = [
    "config",
    "parser",
    "solver",
    "worker",
    "cli",
    "types",
    "api",
    "logging_config",
]

# Public API exports

__api_exports__ = [
    "evaluate",
    "solve_equation",
    "solve_inequality",
    "solve_system",
    "validate_expression",
    "diff",
    "integrate_expr",
    "det",
    "plot",
]
