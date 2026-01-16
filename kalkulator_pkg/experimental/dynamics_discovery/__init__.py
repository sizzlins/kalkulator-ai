"""Dynamics Discovery Module.

Provides SINDy-style algorithms for discovering governing differential
equations from time series data.

Components:
    - sindy: Sparse Identification of Nonlinear Dynamics
    - derivative_estimation: Robust numerical differentiation
"""

from .derivative_estimation import estimate_derivative
from .derivative_estimation import finite_difference
from .derivative_estimation import savgol_derivative
from .derivative_estimation import spectral_derivative
from .derivative_estimation import total_variation_derivative
from .sindy import SINDy
from .sindy import SINDyConfig
from .sindy import discover_ode

__all__ = [
    # Derivative Estimation
    "finite_difference",
    "savgol_derivative",
    "spectral_derivative",
    "total_variation_derivative",
    "estimate_derivative",
    # SINDy
    "SINDy",
    "SINDyConfig",
    "discover_ode",
]
