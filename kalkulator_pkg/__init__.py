"""Kalkulator package: modularized components for parser, solver, worker, and CLI."""

__version__ = "1.4.1"


# Lazy Loading via PEP 562 (Priority 3: Modernization)
import sys
from typing import TYPE_CHECKING

# Eager imports for static analysis only
if TYPE_CHECKING:
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

def __getattr__(name):
    """Lazy load submodules and API functions on access."""
    if name in __all__:
        # If it's a submodule, import it
        if name in ["api", "cli", "config", "logging_config", "parser", "solver", "types", "worker"]:
            import importlib
            return importlib.import_module(f".{name}", __package__)
        
        # If it's a function from api (e.g. evaluate), import api and get it
        if name in ["det", "diff", "evaluate", "integrate_expr", "plot", "solve_equation", 
                   "solve_inequality", "solve_system", "validate_expression"]:
            import importlib
            module = importlib.import_module(".api", __package__)
            return getattr(module, name)
            
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

