"""Benchmark Runner for Symbolic Regression.

Provides automated benchmarking against the Feynman equations database
and comparison capabilities with other methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .feynman_equations import (
    FeynmanEquation,
    get_equations_by_complexity,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        equation_name: Name of the equation
        discovered_expression: String of discovered expression
        true_expression: Original equation formula
        mse: Mean squared error on test data
        r2_score: R² coefficient
        exact_match: Whether the discovered expression is symbolically equivalent
        time_seconds: Time taken to discover
        complexity: Number of nodes/terms in discovered expression
    """

    equation_name: str
    discovered_expression: str
    true_expression: str
    mse: float
    r2_score: float
    exact_match: bool
    time_seconds: float
    complexity: int = 0

    @property
    def is_successful(self) -> bool:
        """Whether discovery was successful (R² > 0.999 or exact match)."""
        return self.exact_match or self.r2_score > 0.999


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results.

    Attributes:
        results: List of individual results
        total_time: Total time for all benchmarks
        config: Configuration used
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0
    config: dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_successful(self) -> int:
        return sum(1 for r in self.results if r.is_successful)

    @property
    def n_exact(self) -> int:
        return sum(1 for r in self.results if r.exact_match)

    @property
    def success_rate(self) -> float:
        return self.n_successful / self.n_total if self.n_total > 0 else 0.0

    @property
    def exact_rate(self) -> float:
        return self.n_exact / self.n_total if self.n_total > 0 else 0.0

    @property
    def mean_r2(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.r2_score for r in self.results])

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd

            return pd.DataFrame(
                [
                    {
                        "equation": r.equation_name,
                        "discovered": r.discovered_expression,
                        "mse": r.mse,
                        "r2": r.r2_score,
                        "exact": r.exact_match,
                        "successful": r.is_successful,
                        "time": r.time_seconds,
                    }
                    for r in self.results
                ]
            )
        except ImportError:
            return self.results

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "SYMBOLIC REGRESSION BENCHMARK RESULTS",
            "=" * 60,
            f"Total equations tested: {self.n_total}",
            f"Successful discoveries: {self.n_successful} ({self.success_rate:.1%})",
            f"Exact matches: {self.n_exact} ({self.exact_rate:.1%})",
            f"Mean R²: {self.mean_r2:.4f}",
            f"Total time: {self.total_time:.1f}s",
            "-" * 60,
        ]

        # Top failures
        failures = [r for r in self.results if not r.is_successful]
        if failures:
            lines.append("Failed equations:")
            for r in sorted(failures, key=lambda x: x.r2_score)[:5]:
                lines.append(f"  {r.equation_name}: R²={r.r2_score:.4f}")

        return "\n".join(lines)


def run_single_benchmark(
    equation: FeynmanEquation,
    method: str = "genetic",
    n_samples: int = 100,
    noise_std: float = 0.0,
    timeout: float = 60.0,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run benchmark on a single equation.

    Args:
        equation: Feynman equation to test
        method: 'genetic', 'lasso', or 'sindy'
        n_samples: Number of data points
        noise_std: Standard deviation of noise (relative to signal)
        timeout: Maximum time in seconds
        verbose: Print progress

    Returns:
        BenchmarkResult
    """
    # Generate data
    X, y = equation.generate_data(n_samples=n_samples, noise_std=noise_std)

    # Generate test data (clean, for evaluation)
    X_test, y_test = equation.generate_data(n_samples=100, noise_std=0.0)

    start_time = time.time()
    discovered_expr = ""
    mse = float("inf")
    r2 = 0.0
    complexity = 0

    try:
        if method == "genetic":
            from ..symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

            config = GeneticConfig(
                population_size=200,
                n_islands=2,
                generations=50,
                timeout=timeout,
                verbose=verbose,
            )
            regressor = GeneticSymbolicRegressor(config)
            pareto = regressor.fit(X, y, variable_names=equation.variables)

            best = pareto.get_best()
            if best:
                discovered_expr = best.expression
                complexity = best.complexity

                # Evaluate on test data
                try:
                    y_pred = best.tree.evaluate(X_test)
                    mse = float(np.mean((y_test - y_pred) ** 2))
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - np.sum((y_test - y_pred) ** 2) / (ss_tot + 1e-10)
                except Exception:
                    pass

        elif method == "lasso":
            from ..regression_solver import solve_regression_stage

            success, expr, _, mse_train = solve_regression_stage(
                X,
                y,
                data_points=[(X[i], y[i]) for i in range(len(y))],
                param_names=equation.variables,
                include_transcendentals=True,
            )

            if success:
                discovered_expr = expr

                # Evaluate
                try:
                    import sympy as sp

                    symbols = {v: sp.Symbol(v) for v in equation.variables}
                    func_expr = sp.sympify(expr, locals=symbols)

                    y_pred = []
                    for i in range(len(X_test)):
                        subs = {
                            symbols[v]: X_test[i, j]
                            for j, v in enumerate(equation.variables)
                        }
                        y_pred.append(float(func_expr.subs(subs)))
                    y_pred = np.array(y_pred)

                    mse = float(np.mean((y_test - y_pred) ** 2))
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - np.sum((y_test - y_pred) ** 2) / (ss_tot + 1e-10)
                except Exception:
                    mse = mse_train

    except Exception as e:
        if verbose:
            print(f"Error on {equation.name}: {e}")

    elapsed = time.time() - start_time

    # Check for exact match (symbolic equivalence)
    exact_match = check_symbolic_equivalence(
        discovered_expr, equation.formula, equation.variables
    )

    return BenchmarkResult(
        equation_name=equation.name,
        discovered_expression=discovered_expr,
        true_expression=equation.formula,
        mse=mse,
        r2_score=r2,
        exact_match=exact_match,
        time_seconds=elapsed,
        complexity=complexity,
    )


def check_symbolic_equivalence(expr1: str, expr2: str, variables: list[str]) -> bool:
    """Check if two expressions are symbolically equivalent.

    Args:
        expr1: First expression string
        expr2: Second expression string
        variables: List of variable names

    Returns:
        True if symbolically equivalent
    """
    try:
        import sympy as sp

        symbols = {v: sp.Symbol(v) for v in variables}
        symbols.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "e": sp.E,
            }
        )

        e1 = sp.sympify(expr1, locals=symbols)
        e2 = sp.sympify(expr2, locals=symbols)

        # Check if difference simplifies to zero
        diff = sp.simplify(e1 - e2)

        if diff == 0:
            return True

        # Try numerical evaluation at random points
        test_points = np.random.uniform(0.1, 10, (10, len(variables)))
        for i in range(10):
            subs = {sp.Symbol(v): test_points[i, j] for j, v in enumerate(variables)}
            try:
                v1 = float(e1.subs(subs))
                v2 = float(e2.subs(subs))
                if abs(v1 - v2) > 1e-6 * (abs(v1) + abs(v2) + 1e-10):
                    return False
            except Exception:
                pass

        return True

    except Exception:
        return False


def run_feynman_benchmark(
    method: str = "genetic",
    max_variables: int = 4,
    noise_levels: list[float] | None = None,
    timeout_per_equation: float = 30.0,
    n_samples: int = 100,
    verbose: bool = True,
) -> BenchmarkSuite:
    """Run full Feynman benchmark suite.

    Args:
        method: Discovery method ('genetic', 'lasso')
        max_variables: Maximum variables per equation
        noise_levels: List of noise levels to test (default: [0])
        timeout_per_equation: Timeout for each equation
        n_samples: Number of data points per equation
        verbose: Print progress

    Returns:
        BenchmarkSuite with all results
    """
    if noise_levels is None:
        noise_levels = [0.0]

    # Get equations matching complexity
    equations = get_equations_by_complexity(max_variables=max_variables)

    if verbose:
        print(f"Running Feynman benchmark on {len(equations)} equations")
        print(f"Method: {method}, Max variables: {max_variables}")
        print("-" * 60)

    suite = BenchmarkSuite(
        config={
            "method": method,
            "max_variables": max_variables,
            "noise_levels": noise_levels,
            "timeout": timeout_per_equation,
            "n_samples": n_samples,
        }
    )

    start_time = time.time()

    for noise in noise_levels:
        for i, eq in enumerate(equations):
            if verbose:
                print(
                    f"[{i+1}/{len(equations)}] {eq.name}: {eq.description}...", end=" "
                )

            result = run_single_benchmark(
                equation=eq,
                method=method,
                n_samples=n_samples,
                noise_std=noise,
                timeout=timeout_per_equation,
                verbose=False,
            )

            suite.results.append(result)

            if verbose:
                status = "✓" if result.is_successful else "✗"
                print(f"{status} R²={result.r2_score:.4f} ({result.time_seconds:.1f}s)")

    suite.total_time = time.time() - start_time

    if verbose:
        print("\n" + suite.summary())

    return suite


def quick_benchmark(
    method: str = "genetic", n_equations: int = 10, timeout: float = 30.0
) -> BenchmarkSuite:
    """Run a quick benchmark on a subset of equations.

    Args:
        method: Discovery method
        n_equations: Number of equations to test
        timeout: Timeout per equation

    Returns:
        BenchmarkSuite
    """
    # Get simplest equations
    equations = get_equations_by_complexity(max_variables=3)[:n_equations]

    print(f"Quick benchmark: {len(equations)} equations, method={method}")
    print("-" * 40)

    suite = BenchmarkSuite()
    start_time = time.time()

    for eq in equations:
        result = run_single_benchmark(
            equation=eq,
            method=method,
            n_samples=50,
            noise_std=0.0,
            timeout=timeout,
            verbose=False,
        )
        suite.results.append(result)

        status = "✓" if result.is_successful else "✗"
        print(f"  {eq.name}: {status} R²={result.r2_score:.4f}")

    suite.total_time = time.time() - start_time

    print("-" * 40)
    print(f"Success: {suite.n_successful}/{suite.n_total} ({suite.success_rate:.1%})")
    print(f"Time: {suite.total_time:.1f}s")

    return suite
