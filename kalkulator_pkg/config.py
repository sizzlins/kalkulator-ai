"""Centralized configuration for Kalkulator.

This module defines:
- Resource limits (CPU time, memory, timeouts)
- Input validation limits (length, depth, node count)
- Cache sizes for performance optimization
- Allowed SymPy functions and transformations
- Regex patterns for parsing
- Solver configuration options

Configuration can be overridden via:
- CLI flags (see cli.py)
- Environment variables (prefixed with KALKULATOR_)
"""

import os
import re

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor
from sympy.parsing.sympy_parser import implicit_multiplication_application
from sympy.parsing.sympy_parser import standard_transformations

from .utils.custom_functions import log2
from .utils.custom_functions import log10

# Version is defined in pyproject.toml [project] section
# Import here for backward compatibility
# try:
#     import importlib.metadata
#
#     VERSION = importlib.metadata.version("kalkulator")
# except Exception:
#     # Fallback if package not installed
VERSION = "1.3.0"

# Resource limits (can be overridden via environment variables)
WORKER_CPU_SECONDS = int(os.getenv("KALKULATOR_WORKER_CPU_SECONDS", "30"))
WORKER_AS_MB = int(os.getenv("KALKULATOR_WORKER_AS_MB", "400"))
WORKER_TIMEOUT = int(os.getenv("KALKULATOR_WORKER_TIMEOUT", "60"))
ENABLE_PERSISTENT_WORKER = (
    os.getenv("KALKULATOR_ENABLE_PERSISTENT_WORKER", "true").lower() == "true"
)
WORKER_POOL_SIZE = int(
    os.getenv("KALKULATOR_WORKER_POOL_SIZE", "4")
)  # Number of parallel worker processes

# Solver configuration
NUMERIC_FALLBACK_ENABLED = (
    os.getenv("KALKULATOR_NUMERIC_FALLBACK_ENABLED", "true").lower() == "true"
)
OUTPUT_PRECISION = int(os.getenv("KALKULATOR_OUTPUT_PRECISION", "6"))
SOLVER_METHOD = os.getenv(
    "KALKULATOR_SOLVER_METHOD", "auto"
)  # "auto", "symbolic", "numeric"

# Input validation limits
MAX_INPUT_LENGTH = int(os.getenv("KALKULATOR_MAX_INPUT_LENGTH", "10000"))  # characters
MAX_EXPRESSION_DEPTH = int(
    os.getenv("KALKULATOR_MAX_EXPRESSION_DEPTH", "100")
)  # tree depth
MAX_EXPRESSION_NODES = int(
    os.getenv("KALKULATOR_MAX_EXPRESSION_NODES", "5000")
)  # total nodes

# Cache configuration
CACHE_SIZE_PARSE = int(os.getenv("KALKULATOR_CACHE_SIZE_PARSE", "1024"))
CACHE_SIZE_EVAL = int(os.getenv("KALKULATOR_CACHE_SIZE_EVAL", "2048"))
CACHE_SIZE_SOLVE = int(os.getenv("KALKULATOR_CACHE_SIZE_SOLVE", "256"))

# Numeric solver configuration
MAX_NSOLVE_GUESSES = int(
    os.getenv("KALKULATOR_MAX_NSOLVE_GUESSES", "50")
)  # Optimized for balance between speed and thoroughness

# Numeric tolerance constants (replacing magic numbers throughout codebase)
NUMERIC_TOLERANCE = float(
    os.getenv("KALKULATOR_NUMERIC_TOLERANCE", "1e-8")
)  # For imaginary part filtering
ROOT_SEARCH_TOLERANCE = float(
    os.getenv("KALKULATOR_ROOT_SEARCH_TOLERANCE", "1e-12")
)  # For root finding precision
MAX_NSOLVE_STEPS = int(
    os.getenv("KALKULATOR_MAX_NSOLVE_STEPS", "80")
)  # Maximum steps for nsolve
COARSE_GRID_MIN_SIZE = int(
    os.getenv("KALKULATOR_COARSE_GRID_MIN_SIZE", "12")
)  # Minimum grid size for root search
ROOT_DEDUP_TOLERANCE = float(
    os.getenv("KALKULATOR_ROOT_DEDUP_TOLERANCE", "1e-6")
)  # For deduplicating roots

# Function finding tolerances and precision
ABSOLUTE_TOLERANCE = float(
    os.getenv("KALKULATOR_ABSOLUTE_TOLERANCE", "1e-10")
)  # Absolute tolerance for exact fits
RELATIVE_TOLERANCE = float(
    os.getenv("KALKULATOR_RELATIVE_TOLERANCE", "1e-8")
)  # Relative tolerance for approximate fits
RESIDUAL_THRESHOLD = float(
    os.getenv("KALKULATOR_RESIDUAL_THRESHOLD", "1e-6")
)  # Threshold for residual checking
CONSTANT_DETECTION_TOLERANCE = float(
    os.getenv("KALKULATOR_CONSTANT_DETECTION_TOLERANCE", "1e-6")
)  # Tolerance for detecting symbolic constants (π, e, etc.)

# Sparse regression configuration
MAX_SUBSET_SEARCH_SIZE = int(
    os.getenv("KALKULATOR_MAX_SUBSET_SEARCH_SIZE", "20")
)  # Maximum size for exhaustive subset search
LASSO_LAMBDA = float(
    os.getenv("KALKULATOR_LASSO_LAMBDA", "0.01")
)  # Default L1 regularization parameter
OMP_MAX_ITERATIONS = int(
    os.getenv("KALKULATOR_OMP_MAX_ITERATIONS", "50")
)  # Maximum iterations for OMP

# Model selection configuration
USE_AIC_BIC = os.getenv("KALKULATOR_USE_AIC_BIC", "true").lower() == "true"
PREFER_SIMPLER_MODELS = (
    os.getenv("KALKULATOR_PREFER_SIMPLER_MODELS", "true").lower() == "true"
)

# ============================================================================
# RESEARCH-GRADE SYMBOLIC REGRESSION CONFIGURATION
# ============================================================================

# Genetic Programming configuration
GP_POPULATION_SIZE = int(
    os.getenv("KALKULATOR_GP_POPULATION_SIZE", "200")
)  # Population per island
GP_GENERATIONS = int(
    os.getenv("KALKULATOR_GP_GENERATIONS", "50")
)  # Maximum generations
GP_PARSIMONY = float(
    os.getenv("KALKULATOR_GP_PARSIMONY", "0.001")
)  # Complexity penalty coefficient
GP_TIMEOUT = float(os.getenv("KALKULATOR_GP_TIMEOUT", "30"))  # Timeout in seconds

# SINDy (Sparse Identification of Nonlinear Dynamics) configuration
SINDY_THRESHOLD = float(
    os.getenv("KALKULATOR_SINDY_THRESHOLD", "0.1")
)  # Sparsity threshold
SINDY_POLY_ORDER = int(
    os.getenv("KALKULATOR_SINDY_POLY_ORDER", "3")
)  # Maximum polynomial order

# Causal Discovery configuration
CAUSAL_ALPHA = float(
    os.getenv("KALKULATOR_CAUSAL_ALPHA", "0.05")
)  # Significance level for independence tests

# Robust regression configuration
ROBUST_METHOD = os.getenv(
    "KALKULATOR_ROBUST_METHOD", "auto"
)  # "auto", "huber", "ransac", "irls"
RANSAC_THRESHOLD = float(
    os.getenv("KALKULATOR_RANSAC_THRESHOLD", "3.0")
)  # MAD multiplier for RANSAC

ALLOWED_SYMPY_NAMES = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    # Traditional math notation aliases (arcsin = asin, etc.)
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    # Use custom classes to ensure proper parsing behavior (lambdas can cause TypeErrors with implicit multiplication)
    "log2": log2,
    "log10": log10,
    "exp": sp.exp,
    "Abs": sp.Abs,
    "abs": sp.Abs,  # lowercase alias for convenience
    # Hyperbolic functions
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "cot": sp.cot,
    # Modulo
    "Mod": sp.Mod,
    "mod": sp.Mod,  # lowercase alias for convenience
    # Calculus & algebra
    "diff": sp.diff,
    "integrate": sp.integrate,
    "limit": sp.limit,  # For evaluating limits: limit(sin(x)/x, x, 0) -> 1
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    # Matrices (basic)
    "Matrix": sp.Matrix,
    "matrix": sp.Matrix,  # lowercase alias for convenience
    "det": sp.det,
    # Special functions
    "LambertW": sp.LambertW,
    "min": sp.Min,
    "max": sp.Max,
    # Factorial and combinatorics
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    # Rounding functions
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "ceil": sp.ceiling,  # alias
    # Number theory
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    # Sign and gamma
    "sign": sp.sign,
    "gamma": sp.gamma,
    # Missing trig functions
    "sec": sp.sec,
    "csc": sp.csc,
    # Inverse hyperbolic functions
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    # Two-argument arctangent
    "atan2": sp.atan2,
    # Roots
    "root": sp.root,
    "cbrt": sp.cbrt,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PERCENT_REGEX = re.compile(r"(\d+(?:\.\d+)?)%")
SQRT_UNICODE_REGEX = re.compile(r"√\s*\(")
DIGIT_LETTERS_REGEX = re.compile(r"(?<![a-zA-Z_])(\d)\s*([A-Za-z(])")
AMBIG_FRACTION_REGEX = re.compile(r"\(([^()]+?)/([^()]+?)\)")
