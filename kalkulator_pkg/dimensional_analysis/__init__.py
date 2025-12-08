"""Dimensional Analysis Module.

Provides unit-aware symbolic regression using dimensional analysis.

Components:
    - units: SI dimensions, Buckingham Pi theorem, dimensionless groups
"""

from .units import (  # Core classes; Common dimensions; Derived dimensions; Functions
    ACCELERATION,
    AMOUNT,
    CHARGE,
    CURRENT,
    DENSITY,
    DIMENSIONLESS,
    ENERGY,
    FORCE,
    FREQUENCY,
    LENGTH,
    LUMINOSITY,
    MASS,
    POWER,
    PRESSURE,
    TEMPERATURE,
    TIME,
    VELOCITY,
    VOLTAGE,
    Dimension,
    Quantity,
    find_dimensionless_groups,
    format_pi_group,
    unit_consistent_features,
)

__all__ = [
    # Classes
    "Dimension",
    "Quantity",
    # Base dimensions
    "DIMENSIONLESS",
    "MASS",
    "LENGTH",
    "TIME",
    "CURRENT",
    "TEMPERATURE",
    "AMOUNT",
    "LUMINOSITY",
    # Derived dimensions
    "VELOCITY",
    "ACCELERATION",
    "FORCE",
    "ENERGY",
    "POWER",
    "FREQUENCY",
    "PRESSURE",
    "DENSITY",
    "CHARGE",
    "VOLTAGE",
    # Functions
    "find_dimensionless_groups",
    "format_pi_group",
    "unit_consistent_features",
]
