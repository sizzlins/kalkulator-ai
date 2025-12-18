"""Kalkulator package: modularized components for parser, solver, worker, and CLI."""

__version__ = "1.0.0"

from . import api, cli, config, logging_config, parser, solver, types, worker
from .api import (
    det,
    diff,
    evaluate,
    integrate_expr,
    plot,
    solve_equation,
    solve_inequality,
    solve_system,
    validate_expression,
)

__all__ = [
    "config",
    "parser",
    "solver",
    "worker",
    "cli",
    "types",
    "api",
    "logging_config",
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
