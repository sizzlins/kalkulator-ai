"""Numerical differentiation methods for time series data.

Provides robust methods for estimating derivatives from noisy
time series data, which is essential for SINDy-style ODE discovery.

Methods:
- Finite differences with various accuracy orders
- Savitzky-Golay smoothed derivatives
- Total variation regularized derivatives
- Spectral differentiation
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def finite_difference(
    x: np.ndarray, t: np.ndarray, order: int = 1, accuracy: int = 2
) -> np.ndarray:
    """Compute derivatives using finite differences.

    Args:
        x: Time series values of shape (n_samples,) or (n_samples, n_vars)
        t: Time points of shape (n_samples,)
        order: Derivative order (1 = first derivative, 2 = second, etc.)
        accuracy: Accuracy order (2, 4, 6 for 2nd, 4th, 6th order accurate)

    Returns:
        Derivative estimates of same shape as x
    """
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_samples, n_vars = x.shape
    dt = np.diff(t)

    # Uniform time step (use mean if slightly non-uniform)
    dt_mean = np.mean(dt)

    derivatives = np.zeros_like(x)

    for v in range(n_vars):
        xv = x[:, v]

        if order == 1:
            if accuracy >= 4 and n_samples >= 5:
                # 4th order accurate central difference
                derivatives[2:-2, v] = (
                    -xv[4:] + 8 * xv[3:-1] - 8 * xv[1:-3] + xv[:-4]
                ) / (12 * dt_mean)
                # Forward/backward at edges
                derivatives[0, v] = (xv[1] - xv[0]) / dt_mean
                derivatives[1, v] = (xv[2] - xv[0]) / (2 * dt_mean)
                derivatives[-2, v] = (xv[-1] - xv[-3]) / (2 * dt_mean)
                derivatives[-1, v] = (xv[-1] - xv[-2]) / dt_mean
            else:
                # 2nd order central difference
                derivatives[1:-1, v] = (xv[2:] - xv[:-2]) / (2 * dt_mean)
                # Forward/backward at edges
                derivatives[0, v] = (xv[1] - xv[0]) / dt_mean
                derivatives[-1, v] = (xv[-1] - xv[-2]) / dt_mean

        elif order == 2:
            if n_samples >= 3:
                # Second derivative
                derivatives[1:-1, v] = (xv[2:] - 2 * xv[1:-1] + xv[:-2]) / (dt_mean**2)
                derivatives[0, v] = derivatives[1, v]
                derivatives[-1, v] = derivatives[-2, v]

    return derivatives if derivatives.shape[1] > 1 else derivatives.ravel()


def savgol_derivative(
    x: np.ndarray,
    t: np.ndarray,
    window_length: int = 5,
    polyorder: int = 3,
    deriv_order: int = 1,
) -> np.ndarray:
    """Compute smoothed derivatives using Savitzky-Golay filter.

    This method applies a polynomial fit in a sliding window, which
    provides both smoothing and differentiation. Better for noisy data.

    Args:
        x: Time series values
        t: Time points
        window_length: Length of the filter window (must be odd)
        polyorder: Order of polynomial to fit
        deriv_order: Derivative order

    Returns:
        Smoothed derivative estimates
    """
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure window_length is valid
    window_length = min(window_length, len(x))
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    # Time step
    dt = np.mean(np.diff(t))

    n_vars = x.shape[1]
    derivatives = np.zeros_like(x)

    for v in range(n_vars):
        xv = x[:, v]

        try:
            # Apply Savitzky-Golay filter with derivative
            deriv = signal.savgol_filter(
                xv,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv_order,
                delta=dt,
            )
            derivatives[:, v] = deriv
        except Exception:
            # Fallback to finite difference
            derivatives[:, v] = finite_difference(xv, t, order=deriv_order).ravel()

    return derivatives if derivatives.shape[1] > 1 else derivatives.ravel()


def spectral_derivative(x: np.ndarray, t: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute derivatives using spectral (FFT) methods.

    Best for periodic or smooth data. Not suitable for non-periodic
    boundary conditions without preprocessing.

    Args:
        x: Time series values (should be periodic-like)
        t: Time points (uniform spacing assumed)
        order: Derivative order

    Returns:
        Derivative estimates
    """
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_samples, n_vars = x.shape
    dt = np.mean(np.diff(t))

    # Frequency array
    freqs = np.fft.fftfreq(n_samples, dt)

    derivatives = np.zeros_like(x)

    for v in range(n_vars):
        xv = x[:, v]

        # FFT
        x_fft = np.fft.fft(xv)

        # Multiply by (i * 2 * pi * f)^order
        deriv_fft = x_fft * (1j * 2 * np.pi * freqs) ** order

        # Inverse FFT
        deriv = np.real(np.fft.ifft(deriv_fft))
        derivatives[:, v] = deriv

    return derivatives if derivatives.shape[1] > 1 else derivatives.ravel()


def total_variation_derivative(
    x: np.ndarray, t: np.ndarray, regularization: float = 1e-2, max_iter: int = 100
) -> np.ndarray:
    """Compute derivatives using total variation regularization.

    Minimizes: ||u' - x'_fd|| + λ * TV(u')
    where TV is the total variation (sum of absolute differences).
    Good for piecewise constant derivatives.

    Args:
        x: Time series values
        t: Time points
        regularization: TV regularization strength
        max_iter: Maximum iterations

    Returns:
        Regularized derivative estimates
    """
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_samples, n_vars = x.shape
    dt = np.mean(np.diff(t))

    derivatives = np.zeros_like(x)

    for v in range(n_vars):
        xv = x[:, v]

        # Initial estimate using finite difference
        u = np.gradient(xv, dt)

        # Iterative proximal gradient descent for TV regularization
        for _ in range(max_iter):
            # Gradient of data fidelity term
            grad = u - np.gradient(xv, dt)

            # Gradient step
            u_temp = u - 0.5 * grad

            # Proximal operator for TV (soft thresholding of differences)
            diff = np.diff(u_temp)
            shrunk = np.sign(diff) * np.maximum(np.abs(diff) - regularization * dt, 0)

            # Reconstruct
            u_new = np.zeros_like(u)
            u_new[0] = u_temp[0]
            u_new[1:] = u_new[0] + np.cumsum(shrunk)

            # Convergence check
            if np.max(np.abs(u_new - u)) < 1e-6:
                u = u_new
                break
            u = u_new

        derivatives[:, v] = u

    return derivatives if derivatives.shape[1] > 1 else derivatives.ravel()


def estimate_derivative(
    x: np.ndarray, t: np.ndarray, method: str = "savgol", order: int = 1, **kwargs
) -> np.ndarray:
    """Estimate derivatives using the specified method.

    Args:
        x: Time series values
        t: Time points
        method: 'finite', 'savgol', 'spectral', or 'tv'
        order: Derivative order
        **kwargs: Additional arguments for specific methods

    Returns:
        Derivative estimates
    """
    methods = {
        "finite": lambda: finite_difference(x, t, order=order, **kwargs),
        "savgol": lambda: savgol_derivative(x, t, deriv_order=order, **kwargs),
        "spectral": lambda: spectral_derivative(x, t, order=order),
        "tv": lambda: total_variation_derivative(x, t, **kwargs),
    }

    method_func = methods.get(method, methods["savgol"])
    return method_func()
