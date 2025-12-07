"""Physical units and dimensional analysis.

Provides:
- Dimension class for representing SI base dimensions
- Quantity class for values with dimensions
- Dimensionless group detection
- Unit-consistent regression
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dimension:
    """SI base dimensions: [M, L, T, I, Θ, N, J].
    
    Represents the dimensional formula of a physical quantity.
    Uses integer exponents for exact representation.
    
    Examples:
        - Force: Dimension(M=1, L=1, T=-2)  # kg·m/s²
        - Energy: Dimension(M=1, L=2, T=-2)  # kg·m²/s²
        - Velocity: Dimension(L=1, T=-1)  # m/s
    
    Attributes:
        M: Mass exponent (kg)
        L: Length exponent (m)
        T: Time exponent (s)
        I: Electric current exponent (A)
        Theta: Temperature exponent (K)
        N: Amount of substance exponent (mol)
        J: Luminous intensity exponent (cd)
    """
    M: int = 0      # Mass (kg)
    L: int = 0      # Length (m)
    T: int = 0      # Time (s)
    I: int = 0      # Current (A)
    Theta: int = 0  # Temperature (K)
    N: int = 0      # Amount (mol)
    J: int = 0      # Luminosity (cd)
    
    def __mul__(self, other: Dimension) -> Dimension:
        """Multiply dimensions (add exponents)."""
        return Dimension(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
            I=self.I + other.I,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )
    
    def __truediv__(self, other: Dimension) -> Dimension:
        """Divide dimensions (subtract exponents)."""
        return Dimension(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )
    
    def __pow__(self, n: int) -> Dimension:
        """Raise dimension to a power (multiply exponents)."""
        return Dimension(
            M=self.M * n,
            L=self.L * n,
            T=self.T * n,
            I=self.I * n,
            Theta=self.Theta * n,
            N=self.N * n,
            J=self.J * n,
        )
    
    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless quantity."""
        return (self.M == 0 and self.L == 0 and self.T == 0 and
                self.I == 0 and self.Theta == 0 and self.N == 0 and self.J == 0)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array of exponents."""
        return np.array([self.M, self.L, self.T, self.I, self.Theta, self.N, self.J])
    
    @staticmethod
    def from_vector(v: np.ndarray) -> Dimension:
        """Create Dimension from exponent vector."""
        v = np.asarray(v).astype(int)
        return Dimension(
            M=int(v[0]), L=int(v[1]), T=int(v[2]), I=int(v[3]),
            Theta=int(v[4]), N=int(v[5]), J=int(v[6])
        )
    
    def __str__(self) -> str:
        parts = []
        names = ['M', 'L', 'T', 'I', 'Θ', 'N', 'J']
        for name, exp in zip(names, self.to_vector()):
            if exp != 0:
                if exp == 1:
                    parts.append(name)
                else:
                    parts.append(f'{name}^{exp}')
        return ' '.join(parts) if parts else '1 (dimensionless)'


# Common dimensions
DIMENSIONLESS = Dimension()
MASS = Dimension(M=1)
LENGTH = Dimension(L=1)
TIME = Dimension(T=1)
CURRENT = Dimension(I=1)
TEMPERATURE = Dimension(Theta=1)
AMOUNT = Dimension(N=1)
LUMINOSITY = Dimension(J=1)

# Derived dimensions
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME**2
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME
FREQUENCY = DIMENSIONLESS / TIME
PRESSURE = FORCE / LENGTH**2
DENSITY = MASS / LENGTH**3
CHARGE = CURRENT * TIME
VOLTAGE = ENERGY / CHARGE


@dataclass
class Quantity:
    """A physical quantity with value and dimension.
    
    Attributes:
        value: Numerical value (scalar or array)
        dimension: Physical dimension
        name: Optional name/label
    """
    value: float | np.ndarray
    dimension: Dimension
    name: str = ""
    
    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                value=self.value * other.value,
                dimension=self.dimension * other.dimension,
            )
        else:
            return Quantity(value=self.value * other, dimension=self.dimension)
    
    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                value=self.value / other.value,
                dimension=self.dimension / other.dimension,
            )
        else:
            return Quantity(value=self.value / other, dimension=self.dimension)
    
    def __pow__(self, n: int):
        return Quantity(
            value=self.value ** n,
            dimension=self.dimension ** n,
        )
    
    def __add__(self, other):
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(f"Cannot add quantities with different dimensions: "
                               f"{self.dimension} and {other.dimension}")
            return Quantity(value=self.value + other.value, dimension=self.dimension)
        raise TypeError("Can only add Quantity to Quantity")
    
    def __sub__(self, other):
        if isinstance(other, Quantity):
            if self.dimension != other.dimension:
                raise ValueError(f"Cannot subtract quantities with different dimensions")
            return Quantity(value=self.value - other.value, dimension=self.dimension)
        raise TypeError("Can only subtract Quantity from Quantity")
    
    def is_dimensionless(self) -> bool:
        return self.dimension.is_dimensionless()
    
    def __str__(self) -> str:
        return f"{self.value} [{self.dimension}]"


def find_dimensionless_groups(
    quantities: list[tuple[str, Dimension]]
) -> list[dict[str, int]]:
    """Apply Buckingham Pi theorem to find dimensionless groups.
    
    Given a set of physical quantities, finds all independent dimensionless
    combinations (Pi groups) that can be formed.
    
    Args:
        quantities: List of (name, Dimension) tuples
        
    Returns:
        List of dicts mapping variable names to their exponents in each Pi group
        
    Example:
        >>> quantities = [
        ...     ('F', FORCE),
        ...     ('rho', DENSITY),
        ...     ('v', VELOCITY),
        ...     ('L', LENGTH),
        ... ]
        >>> groups = find_dimensionless_groups(quantities)
        >>> # Returns something like [{'F': 1, 'rho': -1, 'v': -2, 'L': -2}]
        >>> # Which represents F / (rho * v^2 * L^2) - the drag coefficient!
    """
    names = [q[0] for q in quantities]
    dims = [q[1] for q in quantities]
    
    n_vars = len(quantities)
    
    # Build dimension matrix: each column is a variable, each row is a base dimension
    dim_matrix = np.array([d.to_vector() for d in dims]).T  # Shape: (7, n_vars)
    
    # Find rank of dimension matrix
    rank = np.linalg.matrix_rank(dim_matrix)
    
    # Number of independent Pi groups = n_vars - rank
    n_pi_groups = n_vars - rank
    
    if n_pi_groups <= 0:
        return []
    
    # Find null space of dimension matrix
    # Each null space vector represents exponents for a dimensionless group
    try:
        from scipy.linalg import null_space
        null = null_space(dim_matrix)
    except Exception:
        # Fallback: simple SVD-based null space
        U, s, Vh = np.linalg.svd(dim_matrix)
        null_mask = s < 1e-10
        if np.sum(null_mask) == 0:
            # No null space found, try extending
            null_mask = np.zeros(len(s) + (n_vars - len(s)), dtype=bool)
            null_mask[rank:] = True
        null = Vh[rank:].T  # Null space vectors
    
    # Convert to integer exponents
    pi_groups = []
    for i in range(null.shape[1]):
        vec = null[:, i]
        
        # Try to find integer representation
        # Scale to make smallest non-zero element = 1
        nonzero = np.abs(vec) > 1e-10
        if not np.any(nonzero):
            continue
        
        min_nonzero = np.min(np.abs(vec[nonzero]))
        scaled = vec / min_nonzero
        
        # Round to integers
        int_vec = np.round(scaled).astype(int)
        
        # Verify it's actually null
        check = dim_matrix @ int_vec
        if np.max(np.abs(check)) > 1e-10:
            # Not exactly null, use original
            int_vec = np.round(scaled * 10).astype(int)  # Try scaling up
        
        # Create dict
        group = {}
        for j, name in enumerate(names):
            if int_vec[j] != 0:
                group[name] = int_vec[j]
        
        if group:
            pi_groups.append(group)
    
    return pi_groups


def format_pi_group(group: dict[str, int]) -> str:
    """Format a Pi group as a readable string.
    
    Args:
        group: Dict mapping variable names to exponents
        
    Returns:
        Formatted string like "F / (rho * v^2 * L^2)"
    """
    numerator = []
    denominator = []
    
    for name, exp in group.items():
        if exp > 0:
            if exp == 1:
                numerator.append(name)
            else:
                numerator.append(f"{name}^{exp}")
        else:
            if exp == -1:
                denominator.append(name)
            else:
                denominator.append(f"{name}^{-exp}")
    
    num_str = " * ".join(numerator) if numerator else "1"
    
    if denominator:
        den_str = " * ".join(denominator)
        if len(denominator) > 1:
            return f"{num_str} / ({den_str})"
        else:
            return f"{num_str} / {den_str}"
    else:
        return num_str


def unit_consistent_features(
    X: np.ndarray,
    variable_dimensions: list[Dimension],
    target_dimension: Dimension,
    max_order: int = 2
) -> tuple[np.ndarray, list[str], list[Dimension]]:
    """Generate only dimensionally consistent features for regression.
    
    Args:
        X: Input data of shape (n_samples, n_vars)
        variable_dimensions: Dimension of each input variable
        target_dimension: Required dimension of output
        max_order: Maximum power for each variable
        
    Returns:
        Tuple of (feature_matrix, feature_names, feature_dimensions)
    """
    n_samples, n_vars = X.shape
    
    features = []
    names = []
    dims = []
    
    # Generate all power combinations up to max_order
    def generate_powers(n: int, max_pow: int) -> list[tuple]:
        if n == 0:
            return [()]
        result = []
        for p in range(-max_pow, max_pow + 1):
            for rest in generate_powers(n - 1, max_pow):
                result.append((p,) + rest)
        return result
    
    all_powers = generate_powers(n_vars, max_order)
    
    for powers in all_powers:
        # Calculate resulting dimension
        result_dim = DIMENSIONLESS
        for i, p in enumerate(powers):
            if p != 0:
                result_dim = result_dim * (variable_dimensions[i] ** p)
        
        # Check if it matches target
        if result_dim == target_dimension:
            # Compute feature
            term = np.ones(n_samples)
            name_parts = []
            
            for i, p in enumerate(powers):
                if p != 0:
                    term *= X[:, i] ** p
                    if p == 1:
                        name_parts.append(f'x{i}')
                    else:
                        name_parts.append(f'x{i}^{p}')
            
            if name_parts:
                features.append(term)
                names.append('*'.join(name_parts))
                dims.append(result_dim)
    
    if not features:
        return np.array([]).reshape(n_samples, 0), [], []
    
    return np.column_stack(features), names, dims
