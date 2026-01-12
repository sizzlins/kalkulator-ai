"""Kalkulator package: modularized components for parser, solver, worker, and CLI."""

__version__ = "1.4.1"

from . import api
from . import cli
from . import config
from . import logging_config
from . import parser
from . import solver
from . import types
from . import worker
from .api import det
from .api import diff
from .api import evaluate
from .api import integrate_expr
from .api import plot
from .api import solve_equation
from .api import solve_inequality
from .api import solve_system
from .api import validate_expression

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
