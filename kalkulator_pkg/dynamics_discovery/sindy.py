"""Sparse Identification of Nonlinear Dynamics (SINDy).

SINDy discovers governing differential equations from time series data.
Given measurements x(t), it finds the sparse coefficient matrix Ξ such that:
    dx/dt = Θ(x) * Ξ

where Θ(x) is a library of candidate nonlinear functions.

References:
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
    "Discovering governing equations from data by sparse identification
    of nonlinear dynamical systems." PNAS, 113(15), 3932-3937.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import odeint
from sklearn.linear_model import Lasso, Ridge

from .derivative_estimation import estimate_derivative


@dataclass
class SINDyConfig:
    """Configuration for SINDy algorithm.
    
    Attributes:
        threshold: Sparsity threshold for zeroing small coefficients
        alpha: Regularization strength for sparse regression
        max_iter: Maximum iterations for sequential thresholding
        poly_order: Maximum polynomial order for library
        include_sin: Include sin(x) terms
        include_cos: Include cos(x) terms
        include_exp: Include exp(x) terms (careful with overflow)
        derivative_method: Method for estimating derivatives
        normalize: Whether to normalize the library
    """
    threshold: float = 0.1
    alpha: float = 0.05
    max_iter: int = 10
    poly_order: int = 3
    include_sin: bool = False
    include_cos: bool = False
    include_exp: bool = False
    derivative_method: str = 'savgol'
    normalize: bool = True


class SINDy:
    """Sparse Identification of Nonlinear Dynamics.
    
    Discovers governing ODEs from time series data using sparse regression.
    
    Example:
        >>> # Generate pendulum data
        >>> t = np.linspace(0, 10, 1000)
        >>> theta = 0.5 * np.cos(2 * t)  # Simplified pendulum
        >>> omega = -np.sin(2 * t)
        >>> X = np.column_stack([theta, omega])
        >>> 
        >>> sindy = SINDy()
        >>> sindy.fit(X, t, variable_names=['theta', 'omega'])
        >>> print(sindy.equations)
    """
    
    def __init__(self, config: SINDyConfig | None = None):
        """Initialize SINDy.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or SINDyConfig()
        self.coefficients: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.variable_names: list[str] = []
        self.equations: dict[str, str] = {}
        self._library_functions: list[Callable] = []
    
    def _build_library(
        self,
        X: np.ndarray,
        variable_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Build the library of candidate functions Θ(X).
        
        Args:
            X: State data of shape (n_samples, n_vars)
            variable_names: Names of state variables
            
        Returns:
            Tuple of (library matrix, feature names)
        """
        n_samples, n_vars = X.shape
        features = []
        names = []
        functions = []
        
        # 1. Constant term
        features.append(np.ones(n_samples))
        names.append('1')
        functions.append(lambda x: np.ones(len(x) if hasattr(x, '__len__') else 1))
        
        # 2. Linear terms: x, y, z, ...
        for i, var in enumerate(variable_names):
            features.append(X[:, i])
            names.append(var)
            idx = i  # Capture by value
            functions.append(lambda x, idx=idx: x[idx] if len(x.shape) > 1 else x)
        
        # 3. Polynomial terms up to poly_order
        for order in range(2, self.config.poly_order + 1):
            # Generate all combinations of variables with this total degree
            for combo in self._poly_combinations(n_vars, order):
                term = np.ones(n_samples)
                name_parts = []
                for var_idx, power in enumerate(combo):
                    if power > 0:
                        term *= X[:, var_idx] ** power
                        var_name = variable_names[var_idx]
                        if power == 1:
                            name_parts.append(var_name)
                        else:
                            name_parts.append(f'{var_name}^{power}')
                if name_parts:
                    features.append(term)
                    names.append('*'.join(name_parts))
        
        # 4. Trigonometric terms
        if self.config.include_sin:
            for i, var in enumerate(variable_names):
                features.append(np.sin(X[:, i]))
                names.append(f'sin({var})')
        
        if self.config.include_cos:
            for i, var in enumerate(variable_names):
                features.append(np.cos(X[:, i]))
                names.append(f'cos({var})')
        
        # 5. Exponential terms (use with caution)
        if self.config.include_exp:
            for i, var in enumerate(variable_names):
                exp_term = np.exp(np.clip(X[:, i], -20, 20))  # Prevent overflow
                features.append(exp_term)
                names.append(f'exp({var})')
        
        library = np.column_stack(features)
        self._library_functions = functions
        
        return library, names
    
    def _poly_combinations(self, n_vars: int, total_degree: int) -> list[list[int]]:
        """Generate all polynomial combinations with given total degree.
        
        Args:
            n_vars: Number of variables
            total_degree: Target total degree
            
        Returns:
            List of exponent tuples
        """
        def generate(remaining_degree: int, remaining_vars: int) -> list[list[int]]:
            if remaining_vars == 1:
                return [[remaining_degree]]
            results = []
            for power in range(remaining_degree + 1):
                for rest in generate(remaining_degree - power, remaining_vars - 1):
                    results.append([power] + rest)
            return results
        
        return generate(total_degree, n_vars)
    
    def _sequential_threshold(
        self,
        Theta: np.ndarray,
        dXdt: np.ndarray
    ) -> np.ndarray:
        """Sequential thresholded least squares for sparse regression.
        
        Args:
            Theta: Library matrix of shape (n_samples, n_features)
            dXdt: Derivatives of shape (n_samples, n_vars)
            
        Returns:
            Sparse coefficient matrix of shape (n_features, n_vars)
        """
        n_features = Theta.shape[1]
        n_vars = dXdt.shape[1] if dXdt.ndim > 1 else 1
        
        if dXdt.ndim == 1:
            dXdt = dXdt.reshape(-1, 1)
        
        # Normalize library for better conditioning
        if self.config.normalize:
            norms = np.linalg.norm(Theta, axis=0) + 1e-10
            Theta_norm = Theta / norms
        else:
            norms = np.ones(n_features)
            Theta_norm = Theta
        
        # Initial least squares solution
        Xi = np.linalg.lstsq(Theta_norm, dXdt, rcond=None)[0]
        
        # Sequential thresholding
        for _ in range(self.config.max_iter):
            # Apply threshold
            small_indices = np.abs(Xi) < self.config.threshold
            Xi[small_indices] = 0
            
            # Re-fit with only non-zero terms
            for j in range(n_vars):
                nonzero = ~small_indices[:, j]
                if np.sum(nonzero) > 0:
                    Xi[nonzero, j] = np.linalg.lstsq(
                        Theta_norm[:, nonzero],
                        dXdt[:, j],
                        rcond=None
                    )[0]
        
        # Denormalize
        Xi = Xi / norms.reshape(-1, 1)
        
        return Xi
    
    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        variable_names: list[str] | None = None,
        dXdt: np.ndarray | None = None
    ) -> SINDy:
        """Fit the SINDy model to time series data.
        
        Args:
            X: State data of shape (n_samples,) or (n_samples, n_vars)
            t: Time points of shape (n_samples,)
            variable_names: Names for state variables (default: x0, x1, ...)
            dXdt: Pre-computed derivatives (optional, estimated if None)
            
        Returns:
            self
        """
        X = np.asarray(X)
        t = np.asarray(t)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_vars = X.shape
        
        # Default variable names
        if variable_names is None:
            variable_names = [f'x{i}' for i in range(n_vars)]
        self.variable_names = variable_names
        
        # Estimate derivatives if not provided
        if dXdt is None:
            dXdt = np.zeros_like(X)
            for i in range(n_vars):
                dXdt[:, i] = estimate_derivative(
                    X[:, i], t,
                    method=self.config.derivative_method,
                    order=1
                )
        else:
            dXdt = np.asarray(dXdt)
            if dXdt.ndim == 1:
                dXdt = dXdt.reshape(-1, 1)
        
        # Build library
        Theta, feature_names = self._build_library(X, variable_names)
        self.feature_names = feature_names
        
        # Sparse regression
        self.coefficients = self._sequential_threshold(Theta, dXdt)
        
        # Post-processing: Prune terms that are negligible relative to dominant physics
        # This prevents small noise terms (like 0.02*v^2) from polluting clean laws
        if self.coefficients is not None:
            for j in range(self.coefficients.shape[1]):
                col_coeffs = self.coefficients[:, j]
                max_c = np.max(np.abs(col_coeffs))
                if max_c > 1e-10:
                    # Prune anything less than 10% of the dominant term
                    # This is critical for clean physics discovery
                    col_coeffs[np.abs(col_coeffs) < 0.1 * max_c] = 0
                    self.coefficients[:, j] = col_coeffs
        
        # Build equation strings
        self._build_equations()
        
        return self
    
    def _build_equations(self):
        """Build human-readable equation strings from coefficients."""
        self.equations = {}
        
        if self.coefficients is None:
            return
        
        for var_idx, var_name in enumerate(self.variable_names):
            terms = []
            coefs = self.coefficients[:, var_idx]
            
            for feat_idx, (coef, feat_name) in enumerate(zip(coefs, self.feature_names)):
                if abs(coef) > 1e-10:
                    # Format coefficient
                    if abs(coef - round(coef)) < 1e-4:
                        coef_str = str(int(round(coef)))
                    else:
                        coef_str = f'{coef:.4g}'
                    
                    if feat_name == '1':
                        terms.append(coef_str)
                    elif coef_str == '1':
                        terms.append(feat_name)
                    elif coef_str == '-1':
                        terms.append(f'-{feat_name}')
                    else:
                        terms.append(f'{coef_str}*{feat_name}')
            
            if terms:
                equation = ' + '.join(terms).replace('+ -', '- ')
            else:
                equation = '0'
            
            self.equations[f'd{var_name}/dt'] = equation
    
    def print_equations(self):
        """Print discovered equations."""
        for lhs, rhs in self.equations.items():
            print(f'{lhs} = {rhs}')
    
    def get_sympy_equations(self) -> dict[str, sp.Expr]:
        """Get equations as SymPy expressions.
        
        Returns:
            Dict mapping derivative names to SymPy expressions
        """
        symbols = {var: sp.Symbol(var) for var in self.variable_names}
        sympy_eqs = {}
        
        for lhs, rhs in self.equations.items():
            try:
                expr = sp.sympify(rhs, locals=symbols)
                sympy_eqs[lhs] = sp.simplify(expr)
            except Exception:
                sympy_eqs[lhs] = sp.sympify(rhs)
        
        return sympy_eqs
    
    def simulate(
        self,
        x0: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Simulate the discovered system from initial conditions.
        
        Args:
            x0: Initial state of shape (n_vars,)
            t: Time points for simulation
            
        Returns:
            Simulated states of shape (len(t), n_vars)
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        def dynamics(x, t_val):
            """Right-hand side of the ODE system."""
            # Build library for current state
            x = np.array(x).reshape(1, -1)
            Theta, _ = self._build_library(x, self.variable_names)
            
            # Compute derivatives
            dxdt = (Theta @ self.coefficients).ravel()
            return dxdt
        
        # Integrate
        trajectory = odeint(dynamics, x0, t)
        
        return trajectory
    
    def score(
        self,
        X: np.ndarray,
        t: np.ndarray,
        metric: str = 'r2'
    ) -> float:
        """Score the model on data.
        
        Args:
            X: State data
            t: Time points
            metric: 'r2' or 'mse'
            
        Returns:
            Score value (higher is better for r2, lower for mse)
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Estimate actual derivatives
        dXdt_actual = np.zeros_like(X)
        for i in range(X.shape[1]):
            dXdt_actual[:, i] = estimate_derivative(
                X[:, i], t,
                method=self.config.derivative_method
            )
        
        # Predicted derivatives
        Theta, _ = self._build_library(X, self.variable_names)
        dXdt_pred = Theta @ self.coefficients
        
        if metric == 'mse':
            return float(np.mean((dXdt_actual - dXdt_pred) ** 2))
        else:  # r2
            ss_res = np.sum((dXdt_actual - dXdt_pred) ** 2)
            ss_tot = np.sum((dXdt_actual - np.mean(dXdt_actual, axis=0)) ** 2)
            return float(1 - ss_res / (ss_tot + 1e-10))


def discover_ode(
    X: np.ndarray,
    t: np.ndarray,
    variable_names: list[str] | None = None,
    threshold: float = 0.1,
    poly_order: int = 3,
    include_trig: bool = False
) -> dict[str, str]:
    """Convenience function to discover ODEs from time series data.
    
    Args:
        X: State data
        t: Time points
        variable_names: Variable names
        threshold: Sparsity threshold
        poly_order: Max polynomial order
        include_trig: Include sin/cos terms
        
    Returns:
        Dict of discovered equations
    """
    config = SINDyConfig(
        threshold=threshold,
        poly_order=poly_order,
        include_sin=include_trig,
        include_cos=include_trig,
    )
    
    sindy = SINDy(config)
    sindy.fit(X, t, variable_names)
    
    return sindy.equations
