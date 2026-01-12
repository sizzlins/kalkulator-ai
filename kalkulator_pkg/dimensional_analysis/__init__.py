"""Dimensional Analysis Module.

Provides unit-aware symbolic regression using dimensional analysis.

Components:
    - units: SI dimensions, Buckingham Pi theorem, dimensionless groups
"""

from .units import (
    ACCELERATION,  # Core classes; Common dimensions; Derived dimensions; Functions
)
from .units import AMOUNT
from .units import CHARGE
from .units import CURRENT
from .units import DENSITY
from .units import DIMENSIONLESS
from .units import ENERGY
from .units import FORCE
from .units import FREQUENCY
from .units import LENGTH
from .units import LUMINOSITY
from .units import MASS
from .units import POWER
from .units import PRESSURE
from .units import TEMPERATURE
from .units import TIME
from .units import VELOCITY
from .units import VOLTAGE
from .units import Dimension
from .units import Quantity
from .units import find_dimensionless_groups
from .units import format_pi_group
from .units import unit_consistent_features

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
