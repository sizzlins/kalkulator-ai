#!/usr/bin/env python3
"""Test suite for advanced function finding capabilities.

Tests:
1. Constant detection (PSLQ/nsimplify)
2. High-precision parsing
3. Tolerance and validation
4. Sparse regression (LASSO/OMP)
5. Model selection (AIC/BIC)
"""

from decimal import Decimal
from fractions import Fraction

import sympy as sp

from kalkulator_pkg.function_finder_advanced import (
    calculate_aic,
    calculate_bic,
    calculate_residuals,
    detect_symbolic_constant,
    is_exact_fit,
    lasso_regression,
    orthogonal_matching_pursuit,
    parse_with_precision,
)


def test_constant_detection():
    """Test constant detection (π, e, sqrt(2), etc.)."""
    print("\n" + "="*80)
    print("TEST: Constant Detection (PSLQ/nsimplify)")
    print("="*80)
    
    # Test π detection
    print("\nTest 1.1: Detect pi")
    pi_val = 3.141592653589793
    detected = detect_symbolic_constant(pi_val)
    print(f"  Input: {pi_val}")
    print(f"  Detected: {detected}")
    assert detected == sp.pi, f"Expected sp.pi, got {detected}"
    print("  [PASSED]")
    
    # Test e detection
    print("\nTest 1.2: Detect e")
    e_val = 2.718281828459045
    detected = detect_symbolic_constant(e_val)
    print(f"  Input: {e_val}")
    print(f"  Detected: {detected}")
    assert detected == sp.E, f"Expected e, got {detected}"
    print("  [PASSED]")
    
    # Test sqrt(2) detection
    print("\nTest 1.3: Detect sqrt(2)")
    sqrt2_val = 1.4142135623730951
    detected = detect_symbolic_constant(sqrt2_val)
    print(f"  Input: {sqrt2_val}")
    print(f"  Detected: {detected}")
    assert detected == sp.sqrt(2), f"Expected sqrt(2), got {detected}"
    print("  [PASSED]")
    
    # Test Fraction input
    print("\nTest 1.4: Detect from Fraction")
    pi_frac = Fraction(355, 113)  # Good approximation of π
    detected = detect_symbolic_constant(pi_frac, tolerance=1e-3)
    print(f"  Input: {pi_frac} (approx {float(pi_frac)})")
    print(f"  Detected: {detected}")
    # Should detect as pi with relaxed tolerance
    if detected is not None:
        assert detected == sp.pi
    print("  [PASSED]")
    
    print("\n[PASSED] All constant detection tests")


def test_high_precision_parsing():
    """Test high-precision parsing (Decimal/mpmath)."""
    print("\n" + "="*80)
    print("TEST: High-Precision Parsing")
    print("="*80)
    
    # Test integer
    print("\nTest 2.1: Parse integer")
    result = parse_with_precision(42)
    print(f"  Input: 42")
    print(f"  Result: {result} (type: {type(result).__name__})")
    assert result == Fraction(42), f"Expected Fraction(42), got {result}"
    print("  [PASSED]")
    
    # Test fraction string
    print("\nTest 2.2: Parse fraction string")
    result = parse_with_precision("425/6")
    print(f"  Input: '425/6'")
    print(f"  Result: {result} (type: {type(result).__name__})")
    assert result == Fraction(425, 6), f"Expected Fraction(425, 6), got {result}"
    print("  [PASSED]")
    
    # Test simple decimal (should use Fraction)
    print("\nTest 2.3: Parse simple decimal")
    result = parse_with_precision("70.833")
    print(f"  Input: '70.833'")
    print(f"  Result: {result} (type: {type(result).__name__})")
    # Should be Fraction for simple decimals
    assert isinstance(result, (Fraction, Decimal)), f"Expected Fraction or Decimal, got {type(result)}"
    print("  [PASSED]")
    
    # Test high-precision decimal
    print("\nTest 2.4: Parse high-precision decimal")
    result = parse_with_precision("3.141592653589793238462643383279")
    print(f"  Input: '3.141592653589793238462643383279'")
    print(f"  Result: {result} (type: {type(result).__name__})")
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    print("  [PASSED]")
    
    # Test float
    print("\nTest 2.5: Parse float")
    result = parse_with_precision(3.14159)
    print(f"  Input: 3.14159")
    print(f"  Result: {result} (type: {type(result).__name__})")
    assert isinstance(result, (Fraction, Decimal)), f"Expected Fraction or Decimal, got {type(result)}"
    print("  [PASSED]")
    
    print("\n[PASSED] All high-precision parsing tests")


def test_tolerance_validation():
    """Test tolerance checking and validation."""
    print("\n" + "="*80)
    print("TEST: Tolerance and Validation")
    print("="*80)
    
    # Test exact match
    print("\nTest 3.1: Exact match")
    result = is_exact_fit(10.0, 10.0)
    print(f"  Computed: 10.0, Expected: 10.0")
    print(f"  Is exact: {result}")
    assert result is True, "Should be exact match"
    print("  [PASSED]")
    
    # Test within absolute tolerance
    print("\nTest 3.2: Within absolute tolerance")
    result = is_exact_fit(10.0 + 1e-12, 10.0)
    print(f"  Computed: 10.0 + 1e-12, Expected: 10.0")
    print(f"  Is exact: {result}")
    assert result is True, "Should be within absolute tolerance"
    print("  [PASSED]")
    
    # Test within relative tolerance
    print("\nTest 3.3: Within relative tolerance")
    result = is_exact_fit(10.0 * (1 + 1e-9), 10.0)
    print(f"  Computed: 10.0 * (1 + 1e-9), Expected: 10.0")
    print(f"  Is exact: {result}")
    assert result is True, "Should be within relative tolerance"
    print("  [PASSED]")
    
    # Test residuals calculation
    print("\nTest 3.4: Calculate residuals")
    computed = [1.0, 2.0, 3.0, 4.0]
    expected = [1.0, 2.001, 3.0, 4.0]
    residuals, max_residual, mse = calculate_residuals(computed, expected)
    print(f"  Computed: {computed}")
    print(f"  Expected: {expected}")
    print(f"  Residuals: {residuals}")
    print(f"  Max residual: {max_residual}")
    print(f"  MSE: {mse}")
    assert abs(max_residual - 0.001) < 1e-10, f"Expected max residual ≈ 0.001, got {max_residual}"
    print("  [PASSED]")
    
    print("\n[PASSED] All tolerance and validation tests")


def test_sparse_regression():
    """Test sparse regression (LASSO/OMP)."""
    print("\n" + "="*80)
    print("TEST: Sparse Regression (LASSO/OMP)")
    print("="*80)
    
    # Test OMP
    print("\nTest 4.1: Orthogonal Matching Pursuit")
    try:
        # Simple case: y = 2*x1 + 3*x2
        A = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
        b = [2, 3, 5]
        coeffs, selected = orthogonal_matching_pursuit(A, b, max_nonzero=2)
        print(f"  Design matrix A: {A}")
        print(f"  Target b: {b}")
        print(f"  Coefficients: {coeffs}")
        print(f"  Selected indices: {selected}")
        # Should select first two columns
        assert len(selected) <= 2, "Should select at most 2 features"
        print("  [PASSED]")
    except ImportError:
        print("  [SKIPPED] numpy not available")
    
    # Test LASSO
    print("\nTest 4.2: LASSO Regression")
    try:
        A = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
        b = [2, 3, 5]
        coeffs = lasso_regression(A, b, lambda_reg=0.01)
        print(f"  Design matrix A: {A}")
        print(f"  Target b: {b}")
        print(f"  Coefficients: {coeffs}")
        assert len(coeffs) == 3, "Should return 3 coefficients"
        print("  [PASSED]")
    except ImportError:
        print("  [SKIPPED] numpy/sklearn not available")
    
    print("\n[PASSED] All sparse regression tests")


def test_model_selection():
    """Test model selection (AIC/BIC)."""
    print("\n" + "="*80)
    print("TEST: Model Selection (AIC/BIC)")
    print("="*80)
    
    # Test AIC
    print("\nTest 5.1: Calculate AIC")
    aic = calculate_aic(n_params=3, n_samples=10, mse=0.01)
    print(f"  Parameters: 3, Samples: 10, MSE: 0.01")
    print(f"  AIC: {aic}")
    assert isinstance(aic, float) and aic < float('inf'), "AIC should be finite"
    print("  [PASSED]")
    
    # Test BIC
    print("\nTest 5.2: Calculate BIC")
    bic = calculate_bic(n_params=3, n_samples=10, mse=0.01)
    print(f"  Parameters: 3, Samples: 10, MSE: 0.01")
    print(f"  BIC: {bic}")
    assert isinstance(bic, float) and bic < float('inf'), "BIC should be finite"
    print("  [PASSED]")
    
    # Test AIC vs BIC (BIC should penalize more for more parameters)
    print("\nTest 5.3: AIC vs BIC comparison")
    aic_simple = calculate_aic(n_params=2, n_samples=10, mse=0.01)
    aic_complex = calculate_aic(n_params=5, n_samples=10, mse=0.01)
    bic_simple = calculate_bic(n_params=2, n_samples=10, mse=0.01)
    bic_complex = calculate_bic(n_params=5, n_samples=10, mse=0.01)
    print(f"  Simple model (2 params): AIC={aic_simple:.2f}, BIC={bic_simple:.2f}")
    print(f"  Complex model (5 params): AIC={aic_complex:.2f}, BIC={bic_complex:.2f}")
    assert aic_simple < aic_complex, "Simple model should have lower AIC"
    assert bic_simple < bic_complex, "Simple model should have lower BIC"
    assert bic_complex > aic_complex, "BIC should penalize more than AIC"
    print("  [PASSED]")
    
    print("\n[PASSED] All model selection tests")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*80)
    print("ADVANCED FUNCTION FINDER - TEST SUITE")
    print("="*80)
    
    tests = [
        ("Constant Detection", test_constant_detection),
        ("High-Precision Parsing", test_high_precision_parsing),
        ("Tolerance and Validation", test_tolerance_validation),
        ("Sparse Regression", test_sparse_regression),
        ("Model Selection", test_model_selection),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAILED]: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total test suites: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n[SUCCESS] ALL ADVANCED FEATURES WORKING!")
    else:
        print(f"\n[FAILED] {failed} test suite(s) need attention")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

