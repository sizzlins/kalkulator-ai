"""Numerical Differentiation Utilities for ODE Discovery.

This module provides functions to compute numerical derivatives from
evenly-spaced data, enabling the genetic engine to discover differential
equations like y'' + y = 0 (which defines sin/cos).

Key Functions:
    compute_derivatives: Calculate y, y', y'' from evenly-spaced (x, y) data
    check_even_spacing: Verify data is evenly spaced
"""

import numpy as np


def check_even_spacing(x: np.ndarray, tolerance: float = 1e-6) -> tuple[bool, float]:
    """Check if x values are evenly spaced.
    
    Args:
        x: Array of x values (must be sorted)
        tolerance: Relative tolerance for spacing check
        
    Returns:
        Tuple of (is_evenly_spaced, step_size)
    """
    if len(x) < 2:
        return False, 0.0
    
    diffs = np.diff(x)
    mean_step = np.mean(diffs)
    
    if mean_step == 0:
        return False, 0.0
    
    # Check if all steps are close to the mean
    relative_errors = np.abs(diffs - mean_step) / np.abs(mean_step)
    is_even = np.all(relative_errors < tolerance)
    
    return is_even, float(mean_step)


def compute_derivatives(
    x: np.ndarray, 
    y: np.ndarray,
    validate_spacing: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute y, y', y'' using central differences.
    
    Uses the standard stencils:
        y'[i]  = (y[i+1] - y[i-1]) / (2*dt)        # O(dt²) error
        y''[i] = (y[i+1] - 2*y[i] + y[i-1]) / dt²  # O(dt²) error
    
    Args:
        x: Evenly-spaced x values
        y: Corresponding y values
        validate_spacing: If True, verify x is evenly spaced
        
    Returns:
        Tuple of (x_interior, y_interior, y_prime, y_double_prime)
        Note: Returns interior points only (loses 1 point at each edge)
        
    Raises:
        ValueError: If data is not evenly spaced or insufficient points
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if len(x) < 3:
        raise ValueError(f"Need at least 3 points for 2nd derivative, got {len(x)}")
    
    # Sort by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Check spacing
    if validate_spacing:
        is_even, dt = check_even_spacing(x)
        if not is_even:
            raise ValueError(
                "Data must be evenly spaced for numerical differentiation. "
                "Consider resampling with interpolation."
            )
    else:
        dt = np.mean(np.diff(x))
    
    if dt == 0:
        raise ValueError("Step size cannot be zero")
    
    # Central differences for interior points
    # y'[i] = (y[i+1] - y[i-1]) / (2*dt)
    y_prime = (y[2:] - y[:-2]) / (2 * dt)
    
    # y''[i] = (y[i+1] - 2*y[i] + y[i-1]) / dt²
    y_double_prime = (y[2:] - 2*y[1:-1] + y[:-2]) / (dt**2)
    
    # Return interior points only (indices 1 to n-2)
    x_interior = x[1:-1]
    y_interior = y[1:-1]
    
    return x_interior, y_interior, y_prime, y_double_prime


def resample_to_even_spacing(
    x: np.ndarray, 
    y: np.ndarray, 
    n_points: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Resample unevenly-spaced data to even spacing via linear interpolation.
    
    Args:
        x: Original x values
        y: Original y values
        n_points: Number of output points (default: same as input)
        
    Returns:
        Tuple of (x_even, y_interpolated)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if n_points is None:
        n_points = len(x)
    
    x_even = np.linspace(np.min(x), np.max(x), n_points)
    y_interp = np.interp(x_even, x, y)
    
    return x_even, y_interp
