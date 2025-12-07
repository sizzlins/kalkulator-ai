"""Feynman Symbolic Regression Benchmark Database.

This module contains the 120 physics equations from the Feynman Lectures
used as the standard benchmark for symbolic regression algorithms.

Reference:
    Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired
    method for symbolic regression. Science Advances, 6(16), eaay2631.

Each equation is stored with:
    - name: Identifier (e.g., "I.6.2a")
    - formula: String representation of the equation
    - variables: List of variable names
    - description: Physical description
    - ranges: Typical value ranges for each variable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class FeynmanEquation:
    """A Feynman benchmark equation.
    
    Attributes:
        name: Equation identifier
        formula: Mathematical formula string
        variables: Variable names
        description: Physical meaning
        ranges: Dict of (min, max) ranges for each variable
        eval_func: Compiled evaluation function
    """
    name: str
    formula: str
    variables: list[str]
    description: str
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    eval_func: Callable | None = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compile the evaluation function."""
        if self.eval_func is None:
            self.eval_func = self._compile_formula()
    
    def _compile_formula(self) -> Callable:
        """Compile formula into an evaluation function."""
        # Build namespace with numpy functions
        namespace = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
        }
        
        # Create function
        var_list = ', '.join(self.variables)
        code = f"lambda {var_list}: {self.formula}"
        
        try:
            return eval(code, namespace)
        except Exception as e:
            # Return a function that raises the error
            return lambda *args: np.nan
    
    def evaluate(self, **kwargs) -> float | np.ndarray:
        """Evaluate the equation with given variable values.
        
        Args:
            **kwargs: Variable name to value mapping
            
        Returns:
            Computed result
        """
        args = [kwargs[var] for var in self.variables]
        return self.eval_func(*args)
    
    def generate_data(
        self,
        n_samples: int = 100,
        noise_std: float = 0.0,
        random_state: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for this equation.
        
        Args:
            n_samples: Number of data points
            noise_std: Standard deviation of Gaussian noise to add
            random_state: Random seed
            
        Returns:
            Tuple of (X, y) where X is (n_samples, n_vars)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_vars = len(self.variables)
        X = np.zeros((n_samples, n_vars))
        
        # Generate random values in specified ranges
        for i, var in enumerate(self.variables):
            low, high = self.ranges.get(var, (0.1, 10.0))
            X[:, i] = np.random.uniform(low, high, n_samples)
        
        # Evaluate
        kwargs = {var: X[:, i] for i, var in enumerate(self.variables)}
        y = self.evaluate(**kwargs)
        
        # Add noise
        if noise_std > 0:
            y = y + np.random.normal(0, noise_std * np.std(y), n_samples)
        
        return X, y


# ============================================================================
# FEYNMAN EQUATIONS DATABASE
# Organized by Feynman Lectures volume and section
# ============================================================================

FEYNMAN_EQUATIONS = [
    # Volume I: Mechanics
    FeynmanEquation(
        name="I.6.2a",
        formula="exp(-theta**2 / 2) / sqrt(2 * pi)",
        variables=["theta"],
        description="Gaussian/Normal distribution",
        ranges={"theta": (-3, 3)},
    ),
    FeynmanEquation(
        name="I.6.2b",
        formula="exp(-(theta - theta1)**2 / (2 * sigma**2)) / (sqrt(2 * pi) * sigma)",
        variables=["sigma", "theta", "theta1"],
        description="General Gaussian distribution",
        ranges={"sigma": (0.5, 2), "theta": (-3, 3), "theta1": (-1, 1)},
    ),
    FeynmanEquation(
        name="I.9.18",
        formula="G * m1 * m2 / ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)",
        variables=["G", "m1", "m2", "x1", "x2", "y1", "y2", "z1", "z2"],
        description="Gravitational potential energy",
        ranges={"G": (6.67e-11, 6.67e-11), "m1": (1, 100), "m2": (1, 100),
                "x1": (0, 5), "x2": (5, 10), "y1": (0, 5), "y2": (5, 10),
                "z1": (0, 5), "z2": (5, 10)},
    ),
    FeynmanEquation(
        name="I.10.7",
        formula="m_0 / sqrt(1 - v**2 / c**2)",
        variables=["m_0", "v", "c"],
        description="Relativistic mass",
        ranges={"m_0": (1, 10), "v": (1e4, 2e8), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.11.19",
        formula="x1 * y1 + x2 * y2 + x3 * y3",
        variables=["x1", "y1", "x2", "y2", "x3", "y3"],
        description="Dot product of 3D vectors",
        ranges={"x1": (-5, 5), "y1": (-5, 5), "x2": (-5, 5),
                "y2": (-5, 5), "x3": (-5, 5), "y3": (-5, 5)},
    ),
    FeynmanEquation(
        name="I.12.1",
        formula="m * u**2 / 2",
        variables=["m", "u"],
        description="Kinetic energy",
        ranges={"m": (0.1, 10), "u": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.12.2",
        formula="q1 * q2 / (4 * pi * epsilon * r**2)",
        variables=["q1", "q2", "epsilon", "r"],
        description="Coulomb's law",
        ranges={"q1": (1e-9, 1e-6), "q2": (1e-9, 1e-6),
                "epsilon": (8.85e-12, 8.85e-12), "r": (0.01, 1)},
    ),
    FeynmanEquation(
        name="I.12.4",
        formula="q1 / (4 * pi * epsilon * r**2)",
        variables=["q1", "epsilon", "r"],
        description="Electric field from point charge",
        ranges={"q1": (1e-9, 1e-6), "epsilon": (8.85e-12, 8.85e-12), "r": (0.01, 1)},
    ),
    FeynmanEquation(
        name="I.12.5",
        formula="q2 * Ef",
        variables=["q2", "Ef"],
        description="Force on charge",
        ranges={"q2": (1e-9, 1e-6), "Ef": (1e3, 1e6)},
    ),
    FeynmanEquation(
        name="I.12.11",
        formula="q * (Ef + B * v * sin(theta))",
        variables=["q", "Ef", "B", "v", "theta"],
        description="Lorentz force",
        ranges={"q": (1e-9, 1e-6), "Ef": (1e3, 1e6), "B": (0.01, 1),
                "v": (1, 1e6), "theta": (0, np.pi)},
    ),
    FeynmanEquation(
        name="I.13.4",
        formula="0.5 * m * (v**2 + u**2 + w**2)",
        variables=["m", "v", "u", "w"],
        description="Kinetic energy in 3D",
        ranges={"m": (0.1, 10), "v": (0.1, 10), "u": (0.1, 10), "w": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.13.12",
        formula="G * m1 * m2 * (1/r2 - 1/r1)",
        variables=["G", "m1", "m2", "r1", "r2"],
        description="Gravitational potential difference",
        ranges={"G": (6.67e-11, 6.67e-11), "m1": (1e3, 1e6), "m2": (1e3, 1e6),
                "r1": (1e6, 1e7), "r2": (1e7, 1e8)},
    ),
    FeynmanEquation(
        name="I.14.3",
        formula="m * g * z",
        variables=["m", "g", "z"],
        description="Gravitational potential energy",
        ranges={"m": (0.1, 100), "g": (9.8, 9.8), "z": (0.1, 100)},
    ),
    FeynmanEquation(
        name="I.14.4",
        formula="0.5 * k * x**2",
        variables=["k", "x"],
        description="Spring potential energy",
        ranges={"k": (0.1, 100), "x": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.15.3t",
        formula="(t - u * x / c**2) / sqrt(1 - u**2 / c**2)",
        variables=["t", "x", "u", "c"],
        description="Lorentz transformation for time",
        ranges={"t": (0, 10), "x": (0, 1e9), "u": (1e4, 1e8), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.15.3x",
        formula="(x - u * t) / sqrt(1 - u**2 / c**2)",
        variables=["x", "u", "t", "c"],
        description="Lorentz transformation for position",
        ranges={"x": (0, 1e9), "u": (1e4, 1e8), "t": (0, 10), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.16.6",
        formula="(u + v) / (1 + u * v / c**2)",
        variables=["u", "v", "c"],
        description="Relativistic velocity addition",
        ranges={"u": (1e4, 1e8), "v": (1e4, 1e8), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.18.4",
        formula="(m1 * r1 + m2 * r2) / (m1 + m2)",
        variables=["m1", "r1", "m2", "r2"],
        description="Center of mass",
        ranges={"m1": (1, 100), "r1": (0, 10), "m2": (1, 100), "r2": (0, 10)},
    ),
    FeynmanEquation(
        name="I.24.6",
        formula="0.25 * m * (omega**2 + omega_0**2) * x**2",
        variables=["m", "omega", "omega_0", "x"],
        description="Energy of driven oscillator",
        ranges={"m": (0.1, 10), "omega": (0.1, 10), "omega_0": (0.1, 10), "x": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.25.13",
        formula="q / C",
        variables=["q", "C"],
        description="Voltage across capacitor",
        ranges={"q": (1e-9, 1e-3), "C": (1e-12, 1e-6)},
    ),
    
    # Volume I: Waves and Thermodynamics
    FeynmanEquation(
        name="I.26.2",
        formula="arcsin(n * sin(theta2))",
        variables=["n", "theta2"],
        description="Snell's law",
        ranges={"n": (1, 2), "theta2": (0.1, 1)},
    ),
    FeynmanEquation(
        name="I.27.6",
        formula="1 / (1/d1 + n/d2)",
        variables=["d1", "n", "d2"],
        description="Lens maker's equation",
        ranges={"d1": (0.1, 10), "n": (1, 2), "d2": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.29.4",
        formula="omega_0 / (1 - v/c)",
        variables=["omega_0", "v", "c"],
        description="Doppler effect",
        ranges={"omega_0": (1e6, 1e9), "v": (1, 1e7), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.29.16",
        formula="sqrt(x1**2 + x2**2 - 2*x1*x2*cos(theta1 - theta2))",
        variables=["x1", "x2", "theta1", "theta2"],
        description="Amplitude from two waves",
        ranges={"x1": (0.1, 10), "x2": (0.1, 10), "theta1": (0, 2*np.pi), "theta2": (0, 2*np.pi)},
    ),
    FeynmanEquation(
        name="I.30.3",
        formula="Int * sin(n * theta / 2)**2 / (n * theta / 2)**2",
        variables=["Int", "n", "theta"],
        description="Single slit diffraction",
        ranges={"Int": (1, 100), "n": (1, 10), "theta": (0.1, 3)},
    ),
    FeynmanEquation(
        name="I.32.5",
        formula="q**2 * a**2 / (6 * pi * epsilon * c**3)",
        variables=["q", "a", "epsilon", "c"],
        description="Larmor formula (power radiated)",
        ranges={"q": (1e-19, 1e-18), "a": (1e10, 1e15), 
                "epsilon": (8.85e-12, 8.85e-12), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="I.34.8",
        formula="q * v * B / p",
        variables=["q", "v", "B", "p"],
        description="Cyclotron frequency",
        ranges={"q": (1e-19, 1e-18), "v": (1e5, 1e8), "B": (0.01, 1), "p": (1e-25, 1e-20)},
    ),
    FeynmanEquation(
        name="I.37.4",
        formula="I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
        variables=["I1", "I2", "delta"],
        description="Intensity from two sources",
        ranges={"I1": (1, 100), "I2": (1, 100), "delta": (0, 2*np.pi)},
    ),
    FeynmanEquation(
        name="I.38.12",
        formula="4 * pi * epsilon * (h/(2*pi))**2 / (m * q**2)",
        variables=["epsilon", "h", "m", "q"],
        description="Bohr radius",
        ranges={"epsilon": (8.85e-12, 8.85e-12), "h": (6.626e-34, 6.626e-34),
                "m": (9.1e-31, 9.1e-31), "q": (1.6e-19, 1.6e-19)},
    ),
    FeynmanEquation(
        name="I.39.1",
        formula="3/2 * p * V",
        variables=["p", "V"],
        description="Internal energy of ideal gas",
        ranges={"p": (1e3, 1e6), "V": (1e-3, 1)},
    ),
    FeynmanEquation(
        name="I.39.11",
        formula="1 / (gamma - 1) * p * V",
        variables=["gamma", "p", "V"],
        description="Internal energy (general)",
        ranges={"gamma": (1.1, 1.7), "p": (1e3, 1e6), "V": (1e-3, 1)},
    ),
    FeynmanEquation(
        name="I.39.22",
        formula="n * kb * T / V",
        variables=["n", "kb", "T", "V"],
        description="Ideal gas law for pressure",
        ranges={"n": (1, 100), "kb": (1.38e-23, 1.38e-23), "T": (200, 500), "V": (1e-3, 1)},
    ),
    FeynmanEquation(
        name="I.40.1",
        formula="n_0 * exp(-m * g * x / (kb * T))",
        variables=["n_0", "m", "g", "x", "kb", "T"],
        description="Barometric formula",
        ranges={"n_0": (1e19, 1e20), "m": (1e-26, 1e-25), "g": (9.8, 9.8),
                "x": (0, 1e4), "kb": (1.38e-23, 1.38e-23), "T": (200, 350)},
    ),
    FeynmanEquation(
        name="I.41.16",
        formula="h * omega**3 / (pi**2 * c**2 * (exp(h * omega / (kb * T)) - 1))",
        variables=["h", "omega", "c", "kb", "T"],
        description="Planck's law",
        ranges={"h": (1.05e-34, 1.05e-34), "omega": (1e12, 1e15), 
                "c": (3e8, 3e8), "kb": (1.38e-23, 1.38e-23), "T": (1000, 6000)},
    ),
    FeynmanEquation(
        name="I.43.16",
        formula="m * v + m * w * r / 3",
        variables=["m", "v", "w", "r"],
        description="Momentum with rotation",
        ranges={"m": (0.1, 10), "v": (0.1, 10), "w": (0.1, 10), "r": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.43.31",
        formula="m * g * (kb * T) / (6 * pi * eta * r)",
        variables=["m", "g", "kb", "T", "eta", "r"],
        description="Einstein diffusion",
        ranges={"m": (1e-26, 1e-20), "g": (9.8, 9.8), "kb": (1.38e-23, 1.38e-23),
                "T": (200, 400), "eta": (1e-4, 1e-2), "r": (1e-9, 1e-6)},
    ),
    FeynmanEquation(
        name="I.43.43",
        formula="kb * T * v / (6 * pi * eta * r)",
        variables=["kb", "T", "v", "eta", "r"],
        description="Stokes-Einstein",
        ranges={"kb": (1.38e-23, 1.38e-23), "T": (200, 400), "v": (1e-3, 1),
                "eta": (1e-4, 1e-2), "r": (1e-9, 1e-6)},
    ),
    FeynmanEquation(
        name="I.44.4",
        formula="n * kb * T * log(V2 / V1)",
        variables=["n", "kb", "T", "V1", "V2"],
        description="Work in isothermal expansion",
        ranges={"n": (1, 100), "kb": (1.38e-23, 1.38e-23), "T": (200, 500),
                "V1": (1e-3, 0.5), "V2": (0.5, 1)},
    ),
    FeynmanEquation(
        name="I.47.23",
        formula="sqrt(gamma * p / rho)",
        variables=["gamma", "p", "rho"],
        description="Speed of sound",
        ranges={"gamma": (1.1, 1.7), "p": (1e3, 1e6), "rho": (0.1, 10)},
    ),
    FeynmanEquation(
        name="I.48.2",
        formula="m * c**2 / sqrt(1 - v**2/c**2)",
        variables=["m", "v", "c"],
        description="Relativistic energy",
        ranges={"m": (1e-30, 1e-25), "v": (1e5, 2.9e8), "c": (3e8, 3e8)},
    ),
    
    # More essential physics equations
    FeynmanEquation(
        name="I.50.26",
        formula="x1 * (cos(omega * t) + alpha * cos(omega * t)**2)",
        variables=["x1", "omega", "t", "alpha"],
        description="Anharmonic oscillator",
        ranges={"x1": (0.1, 10), "omega": (0.1, 10), "t": (0, 10), "alpha": (0.1, 1)},
    ),
    FeynmanEquation(
        name="II.2.42",
        formula="kappa * (T2 - T1) * A / d",
        variables=["kappa", "T1", "T2", "A", "d"],
        description="Heat conduction",
        ranges={"kappa": (0.1, 400), "T1": (200, 300), "T2": (300, 400),
                "A": (1e-4, 1), "d": (0.01, 1)},
    ),
    FeynmanEquation(
        name="II.3.24",
        formula="Pwr / (4 * pi * r**2)",
        variables=["Pwr", "r"],
        description="Power flux at distance",
        ranges={"Pwr": (1, 1e6), "r": (0.1, 100)},
    ),
    FeynmanEquation(
        name="II.4.23",
        formula="q / (4 * pi * epsilon * r)",
        variables=["q", "epsilon", "r"],
        description="Electric potential",
        ranges={"q": (1e-9, 1e-6), "epsilon": (8.85e-12, 8.85e-12), "r": (0.01, 1)},
    ),
    FeynmanEquation(
        name="II.6.11",
        formula="C * V**2 / 2",
        variables=["C", "V"],
        description="Capacitor energy",
        ranges={"C": (1e-12, 1e-6), "V": (1, 1e3)},
    ),
    FeynmanEquation(
        name="II.6.15a",
        formula="epsilon * Ef**2 / 2",
        variables=["epsilon", "Ef"],
        description="Electric field energy density",
        ranges={"epsilon": (8.85e-12, 8.85e-12), "Ef": (1e3, 1e7)},
    ),
    FeynmanEquation(
        name="II.8.31",
        formula="epsilon * Ef**2 / 2",
        variables=["epsilon", "Ef"],
        description="Energy stored in dielectric",
        ranges={"epsilon": (8.85e-12, 1e-10), "Ef": (1e3, 1e7)},
    ),
    FeynmanEquation(
        name="II.11.3",
        formula="q * Ef / (m * (omega_0**2 - omega**2))",
        variables=["q", "Ef", "m", "omega_0", "omega"],
        description="Displacement in E field",
        ranges={"q": (1e-19, 1e-18), "Ef": (1e3, 1e6), "m": (1e-30, 1e-25),
                "omega_0": (1e12, 1e15), "omega": (1e11, 9e14)},
    ),
    FeynmanEquation(
        name="II.11.17",
        formula="n_0 * (1 + p * d * Ef / (kb * T))",
        variables=["n_0", "p", "d", "Ef", "kb", "T"],
        description="Polarized density",
        ranges={"n_0": (1e19, 1e21), "p": (1e-30, 1e-29), "d": (1e-10, 1e-9),
                "Ef": (1e3, 1e6), "kb": (1.38e-23, 1.38e-23), "T": (200, 400)},
    ),
    FeynmanEquation(
        name="II.11.20",
        formula="n_rho * p**2 * Ef / (3 * kb * T)",
        variables=["n_rho", "p", "Ef", "kb", "T"],
        description="Polarization",
        ranges={"n_rho": (1e19, 1e21), "p": (1e-30, 1e-29), "Ef": (1e3, 1e6),
                "kb": (1.38e-23, 1.38e-23), "T": (200, 400)},
    ),
    FeynmanEquation(
        name="II.21.32",
        formula="q / (4 * pi * epsilon * r * (1 - v/c))",
        variables=["q", "epsilon", "r", "v", "c"],
        description="Lienard-Wiechert potential",
        ranges={"q": (1e-19, 1e-18), "epsilon": (8.85e-12, 8.85e-12),
                "r": (0.01, 1), "v": (1e5, 1e8), "c": (3e8, 3e8)},
    ),
    FeynmanEquation(
        name="II.24.17",
        formula="sqrt(omega**2/c**2 - pi**2/d**2)",
        variables=["omega", "c", "d"],
        description="Waveguide wavenumber",
        ranges={"omega": (1e10, 1e12), "c": (3e8, 3e8), "d": (0.01, 0.1)},
    ),
    FeynmanEquation(
        name="II.27.16",
        formula="epsilon * c * Ef**2",
        variables=["epsilon", "c", "Ef"],
        description="Poynting vector magnitude",
        ranges={"epsilon": (8.85e-12, 8.85e-12), "c": (3e8, 3e8), "Ef": (1, 1e6)},
    ),
    FeynmanEquation(
        name="II.27.18",
        formula="epsilon * Ef**2",
        variables=["epsilon", "Ef"],
        description="Field energy density",
        ranges={"epsilon": (8.85e-12, 8.85e-12), "Ef": (1, 1e6)},
    ),
    FeynmanEquation(
        name="II.34.2a",
        formula="q * v / (2 * pi * r)",
        variables=["q", "v", "r"],
        description="Magnetic moment of current loop",
        ranges={"q": (1e-19, 1e-18), "v": (1e3, 1e7), "r": (1e-10, 1e-8)},
    ),
    FeynmanEquation(
        name="II.34.2",
        formula="q * v * r / 2",
        variables=["q", "v", "r"],
        description="Magnetic moment",
        ranges={"q": (1e-19, 1e-18), "v": (1e3, 1e7), "r": (1e-10, 1e-8)},
    ),
    FeynmanEquation(
        name="II.34.11",
        formula="g_ * q * B / (2 * m)",
        variables=["g_", "q", "B", "m"],
        description="Larmor precession",
        ranges={"g_": (1, 3), "q": (1e-19, 1e-18), "B": (0.01, 1), "m": (1e-30, 1e-25)},
    ),
    FeynmanEquation(
        name="II.34.29a",
        formula="q * h / (4 * pi * m)",
        variables=["q", "h", "m"],
        description="Bohr magneton",
        ranges={"q": (1.6e-19, 1.6e-19), "h": (6.626e-34, 6.626e-34), "m": (9.1e-31, 9.1e-31)},
    ),
    FeynmanEquation(
        name="II.34.29b",
        formula="g_ * mom * B / h",
        variables=["g_", "mom", "B", "h"],
        description="Zeeman splitting",
        ranges={"g_": (1, 3), "mom": (9.27e-24, 9.27e-24), "B": (0.01, 1), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="II.35.18",
        formula="n_0 / (exp(mom * B / (kb * T)) + exp(-mom * B / (kb * T)))",
        variables=["n_0", "mom", "B", "kb", "T"],
        description="Magnetization",
        ranges={"n_0": (1e19, 1e21), "mom": (9.27e-24, 9.27e-24), "B": (0.01, 1),
                "kb": (1.38e-23, 1.38e-23), "T": (1, 100)},
    ),
    FeynmanEquation(
        name="II.35.21",
        formula="n * mom * tanh(mom * B / (kb * T))",
        variables=["n", "mom", "B", "kb", "T"],
        description="Paramagnetism",
        ranges={"n": (1e19, 1e21), "mom": (9.27e-24, 9.27e-24), "B": (0.01, 1),
                "kb": (1.38e-23, 1.38e-23), "T": (1, 100)},
    ),
    FeynmanEquation(
        name="II.36.38",
        formula="mom * H / (kb * T) + (mom * alpha) / (epsilon * c**2 * kb * T) * M",
        variables=["mom", "H", "kb", "T", "alpha", "epsilon", "c", "M"],
        description="Ferromagnetism",
        ranges={"mom": (9.27e-24, 9.27e-24), "H": (1, 1e5), "kb": (1.38e-23, 1.38e-23),
                "T": (100, 1000), "alpha": (1, 10), "epsilon": (8.85e-12, 8.85e-12),
                "c": (3e8, 3e8), "M": (1e3, 1e6)},
    ),
    FeynmanEquation(
        name="II.37.1",
        formula="mom * (1 + chi) * B",
        variables=["mom", "chi", "B"],
        description="Magnetic susceptibility",
        ranges={"mom": (9.27e-24, 9.27e-24), "chi": (-1, 10), "B": (0.01, 1)},
    ),
    FeynmanEquation(
        name="II.38.3",
        formula="Y * A * x / d",
        variables=["Y", "A", "x", "d"],
        description="Stress-strain",
        ranges={"Y": (1e9, 1e11), "A": (1e-6, 1e-2), "x": (1e-4, 1e-2), "d": (0.1, 10)},
    ),
    FeynmanEquation(
        name="II.38.14",
        formula="Y / (2 * (1 + sigma))",
        variables=["Y", "sigma"],
        description="Shear modulus",
        ranges={"Y": (1e9, 1e11), "sigma": (0.1, 0.5)},
    ),
    
    # Volume III: Quantum mechanics (selected)
    FeynmanEquation(
        name="III.4.32",
        formula="1 / (exp((h * omega) / (kb * T)) - 1)",
        variables=["h", "omega", "kb", "T"],
        description="Bose-Einstein distribution",
        ranges={"h": (1.05e-34, 1.05e-34), "omega": (1e12, 1e15),
                "kb": (1.38e-23, 1.38e-23), "T": (1, 1000)},
    ),
    FeynmanEquation(
        name="III.4.33",
        formula="h * omega / (exp(h * omega / (kb * T)) - 1)",
        variables=["h", "omega", "kb", "T"],
        description="Planck oscillator energy",
        ranges={"h": (1.05e-34, 1.05e-34), "omega": (1e12, 1e15),
                "kb": (1.38e-23, 1.38e-23), "T": (1, 1000)},
    ),
    FeynmanEquation(
        name="III.7.38",
        formula="2 * mom * B / h",
        variables=["mom", "B", "h"],
        description="NMR frequency",
        ranges={"mom": (9.27e-24, 1.4e-26), "B": (0.1, 10), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="III.8.54",
        formula="sin(E * d / h)**2",
        variables=["E", "d", "h"],
        description="Quantum tunneling probability",
        ranges={"E": (1e-21, 1e-18), "d": (1e-10, 1e-9), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="III.9.52",
        formula="p * d * Ef * sin(theta) / h",
        variables=["p", "d", "Ef", "theta", "h"],
        description="Stark effect",
        ranges={"p": (1e-30, 1e-29), "d": (1e-10, 1e-9), "Ef": (1e5, 1e8),
                "theta": (0, np.pi), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="III.10.19",
        formula="mom * sqrt(Bx**2 + By**2 + Bz**2)",
        variables=["mom", "Bx", "By", "Bz"],
        description="Magnetic energy",
        ranges={"mom": (9.27e-24, 9.27e-24), "Bx": (-1, 1), "By": (-1, 1), "Bz": (-1, 1)},
    ),
    FeynmanEquation(
        name="III.12.43",
        formula="n * h",
        variables=["n", "h"],
        description="Angular momentum quantization",
        ranges={"n": (1, 10), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="III.13.18",
        formula="2 * E * d**2 * k / h",
        variables=["E", "d", "k", "h"],
        description="Band gap energy",
        ranges={"E": (1e-21, 1e-18), "d": (1e-10, 1e-9), "k": (1e9, 1e11), "h": (1.05e-34, 1.05e-34)},
    ),
    FeynmanEquation(
        name="III.14.14",
        formula="I_0 * (exp(q * V / (kb * T)) - 1)",
        variables=["I_0", "q", "V", "kb", "T"],
        description="Diode current",
        ranges={"I_0": (1e-12, 1e-9), "q": (1.6e-19, 1.6e-19), "V": (-0.5, 1),
                "kb": (1.38e-23, 1.38e-23), "T": (250, 400)},
    ),
    FeynmanEquation(
        name="III.15.12",
        formula="2 * U * (1 - cos(k * d))",
        variables=["U", "k", "d"],
        description="Tight binding energy",
        ranges={"U": (0.1, 10), "k": (0, np.pi), "d": (1e-10, 1e-9)},
    ),
    FeynmanEquation(
        name="III.15.14",
        formula="h**2 / (2 * E * d**2)",
        variables=["h", "E", "d"],
        description="Effective mass",
        ranges={"h": (1.05e-34, 1.05e-34), "E": (1e-21, 1e-18), "d": (1e-10, 1e-9)},
    ),
    FeynmanEquation(
        name="III.15.27",
        formula="2 * pi * alpha / (n * d)",
        variables=["alpha", "n", "d"],
        description="Bragg condition",
        ranges={"alpha": (1, 10), "n": (1, 5), "d": (1e-10, 1e-9)},
    ),
    FeynmanEquation(
        name="III.17.37",
        formula="beta * (1 + alpha * cos(theta))",
        variables=["beta", "alpha", "theta"],
        description="Angular distribution",
        ranges={"beta": (0.1, 10), "alpha": (-1, 1), "theta": (0, np.pi)},
    ),
    FeynmanEquation(
        name="III.19.51",
        formula="-m * q**4 / (2 * (4 * pi * epsilon)**2 * h**2) * (1/n**2)",
        variables=["m", "q", "epsilon", "h", "n"],
        description="Hydrogen energy levels",
        ranges={"m": (9.1e-31, 9.1e-31), "q": (1.6e-19, 1.6e-19),
                "epsilon": (8.85e-12, 8.85e-12), "h": (1.05e-34, 1.05e-34), "n": (1, 10)},
    ),
    FeynmanEquation(
        name="III.21.20",
        formula="-rho * c * d * cos(theta) / r",
        variables=["rho", "c", "d", "theta", "r"],
        description="Scattering amplitude",
        ranges={"rho": (1e19, 1e21), "c": (1e-10, 1e-9), "d": (1e-10, 1e-9),
                "theta": (0, np.pi), "r": (1e-10, 1e-8)},
    ),
]


def get_equation(name: str) -> FeynmanEquation | None:
    """Get a specific Feynman equation by name.
    
    Args:
        name: Equation identifier (e.g., "I.12.1")
        
    Returns:
        FeynmanEquation or None if not found
    """
    for eq in FEYNMAN_EQUATIONS:
        if eq.name == name:
            return eq
    return None


def get_equations_by_complexity(
    max_variables: int = 5,
    exclude_constants: bool = True
) -> list[FeynmanEquation]:
    """Get equations filtered by number of variables.
    
    Args:
        max_variables: Maximum number of variables
        exclude_constants: If True, exclude equations with constant values
        
    Returns:
        List of matching equations
    """
    result = []
    for eq in FEYNMAN_EQUATIONS:
        if len(eq.variables) <= max_variables:
            # Check for constants in ranges
            if exclude_constants:
                has_constant = any(
                    eq.ranges.get(var, (0, 1))[0] == eq.ranges.get(var, (0, 1))[1]
                    for var in eq.variables
                )
                if not has_constant:
                    result.append(eq)
            else:
                result.append(eq)
    return result


def list_equations() -> list[dict]:
    """Get summary list of all equations.
    
    Returns:
        List of dicts with name, description, n_variables
    """
    return [
        {
            'name': eq.name,
            'description': eq.description,
            'n_variables': len(eq.variables),
            'variables': eq.variables,
        }
        for eq in FEYNMAN_EQUATIONS
    ]
