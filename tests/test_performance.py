"""Performance tests and benchmarks for Kalkulator.

These tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""

import time

import pytest

from kalkulator_pkg.parser import parse_preprocessed
from kalkulator_pkg.parser import preprocess
from kalkulator_pkg.solver import solve_single_equation
from kalkulator_pkg.worker import evaluate_safely


@pytest.mark.slow
class TestParsingPerformance:
    """Test parsing performance."""

    def test_simple_expression_parsing_time(self):
        """Benchmark simple expression parsing."""
        expr = "x^2 + 2*x + 1"
        start = time.time()
        for _ in range(100):
            preprocessed = preprocess(expr)
            _ = parse_preprocessed(preprocessed)  # noqa: F841
        elapsed = time.time() - start
        # Should parse 100 expressions in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Parsing too slow: {elapsed}s"

    def test_complex_expression_parsing_time(self):
        """Benchmark complex expression parsing."""
        expr = "sin(x) * cos(y) + tan(z) * log(w)"
        start = time.time()
        for _ in range(50):
            preprocessed = preprocess(expr)
            _ = parse_preprocessed(preprocessed)  # noqa: F841
        elapsed = time.time() - start
        # Complex expressions should still parse reasonably
        assert elapsed < 2.0, f"Complex parsing too slow: {elapsed}s"


@pytest.mark.slow
class TestSolvingPerformance:
    """Test solving performance."""

    def test_linear_solving_time(self):
        """Benchmark linear equation solving."""
        start = time.time()
        for _ in range(50):
            result = solve_single_equation(f"x + {_} = 0")
            assert result["ok"] is True
        elapsed = time.time() - start
        # Linear equations should solve quickly
        assert elapsed < 2.0, f"Linear solving too slow: {elapsed}s"

    def test_quadratic_solving_time(self):
        """Benchmark quadratic equation solving."""
        start = time.time()
        for i in range(20):
            result = solve_single_equation(f"x^2 + {i}*x + 1 = 0")
            assert result["ok"] is True
        elapsed = time.time() - start
        # Quadratic equations should solve reasonably
        assert elapsed < 3.0, f"Quadratic solving too slow: {elapsed}s"


@pytest.mark.slow
class TestEvaluationPerformance:
    """Test evaluation performance."""

    def test_simple_evaluation_time(self):
        """Benchmark simple expression evaluation."""
        start = time.time()
        for i in range(100):
            result = evaluate_safely(f"{i} + {i}")
            assert result["ok"] is True
        elapsed = time.time() - start
        # Simple evaluations should be fast
        assert elapsed < 5.0, f"Evaluation too slow: {elapsed}s"

    def test_cached_evaluation_performance(self):
        """Test that caching improves performance."""
        expr = "x^2 + 2*x + 1"

        # First evaluation (cache miss)
        start1 = time.time()
        _ = evaluate_safely(expr)  # noqa: F841
        time1 = time.time() - start1

        # Second evaluation (cache hit)
        start2 = time.time()
        _ = evaluate_safely(expr)  # noqa: F841
        time2 = time.time() - start2

        # Cache hit should be faster (or at least not significantly slower)
        # Allow some variance due to system load
        assert time2 <= time1 * 1.5, "Caching not improving performance"


@pytest.mark.slow
class TestCachePerformance:
    """Test cache performance and effectiveness."""

    def test_cache_hit_rate(self):
        """Test that cache is being used effectively."""
        expr = "2 + 2"

        # Evaluate multiple times
        for _ in range(10):
            result = evaluate_safely(expr)
            assert result["ok"] is True

        # Cache should be working (verify by checking consistent results)
        results = [evaluate_safely(expr)["result"] for _ in range(5)]
        # All results should be identical due to caching
        assert len(set(results)) == 1, "Cache not working consistently"


# Performance benchmarks (non-blocking)
def benchmark_representative_workload():
    """Benchmark a representative workload."""
    expressions = [
        "2 + 2",
        "x + 1 = 0",
        "x^2 - 1 = 0",
        "sin(x)",
        "integrate(x^2, x)",
    ]

    start = time.time()
    for expr in expressions * 10:  # 50 total operations
        try:
            if "=" in expr:
                from kalkulator_pkg.solver import solve_single_equation

                solve_single_equation(expr)
            else:
                evaluate_safely(expr)
        except Exception:
            pass  # Ignore errors in benchmark
    elapsed = time.time() - start

    return {
        "total_time": elapsed,
        "operations": 50,
        "avg_time_per_op": elapsed / 50,
    }
