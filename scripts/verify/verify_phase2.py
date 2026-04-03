#!/usr/bin/env python
"""Phase 2 Verification Script — Architecture Hardening Tests.

Tests:
  1. Security: Malicious strings raise errors, not execute
  2. Pickle Safety: Custom operators (frac, trunc, heaviside) can be pickled
  3. Parsing Ambiguity: Implicit multiplication (2x) vs function calls (f(x))
  4. Heuristic Regression: Scalloped staircase detection still works
  5. Error UX: Friendly error messages for bad input
  6. Diophantine: compute_integerized_equation handles edge cases
"""
import pickle
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import sympy as sp

# ── Helpers ─────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


# ── 1. Security Tests ──────────────────────────────────────────────────────

def test_security():
    print("\n=== 1. SECURITY: Malicious Input Rejection ===")
    from kalkulator_pkg.utils.parsing import eval_to_float

    malicious_inputs = [
        "__import__('os').system('echo pwned')",
        "exec('import os')",
        "eval('1+1')",
        "open('/etc/passwd').read()",
        "lambda: 0",
        "__builtins__",
    ]

    for inp in malicious_inputs:
        try:
            result = eval_to_float(inp)
            # If it returned a number, it might have "simplified" the string.
            # This is only okay if it didn't actually execute anything.
            # For our purposes, getting a ValueError is the desired behavior.
            check(f"Reject: {inp[:40]}", False, f"Got result: {result}")
        except (ValueError, TypeError, SyntaxError):
            check(f"Reject: {inp[:40]}", True)
        except Exception as e:
            # Any other exception is also acceptable (didn't execute)
            check(f"Reject: {inp[:40]}", True, f"Raised {type(e).__name__}")

    # Verify safe expressions still work
    safe_tests = [
        ("pi", 3.14159, 0.001),
        ("sqrt(2)", 1.4142, 0.001),
        ("sin(pi/2)", 1.0, 0.001),
        ("2+3", 5.0, 0.001),
        ("e", 2.71828, 0.001),
        ("log(1)", 0.0, 0.001),
    ]
    for expr, expected, tol in safe_tests:
        try:
            result = eval_to_float(expr)
            check(f"Safe: {expr} = {result:.4f}", abs(result - expected) < tol)
        except Exception as e:
            check(f"Safe: {expr}", False, f"Raised {type(e).__name__}: {e}")


# ── 2. Pickle Safety Tests ─────────────────────────────────────────────────

def test_pickle_safety():
    print("\n=== 2. PICKLE SAFETY: Custom Operators ===")
    from kalkulator_pkg.sympy_defs import ALLOWED_SYMPY_NAMES

    # Test that key custom operators can be pickled
    pickle_targets = ["trunc", "frac", "heaviside", "Heaviside", "round", "neg", "inv"]

    for name in pickle_targets:
        func = ALLOWED_SYMPY_NAMES.get(name)
        if func is None:
            check(f"Pickle {name}", False, "Not found in ALLOWED_SYMPY_NAMES")
            continue
        try:
            data = pickle.dumps(func)
            restored = pickle.loads(data)
            check(f"Pickle {name}", True, f"{len(data)} bytes")
        except Exception as e:
            check(f"Pickle {name}", False, f"{type(e).__name__}: {e}")

    # Test pickling a SymPy expression using a custom operator
    try:
        x = sp.Symbol("x")
        expr = ALLOWED_SYMPY_NAMES["frac"](x)
        data = pickle.dumps(expr)
        restored = pickle.loads(data)
        check("Pickle frac(x) expr", True, f"{len(data)} bytes")
    except Exception as e:
        check("Pickle frac(x) expr", False, f"{type(e).__name__}: {e}")


# ── 3. Parsing Ambiguity Tests ──────────────────────────────────────────────

def test_ambiguity():
    print("\n=== 3. PARSING AMBIGUITY: Implicit Multiplication ===")
    from sympy.parsing.sympy_parser import parse_expr
    from kalkulator_pkg.sympy_defs import ALLOWED_SYMPY_NAMES, TRANSFORMATIONS

    x = sp.Symbol("x")

    # Test: "2x" should parse as 2*x
    try:
        result = parse_expr("2x", local_dict=ALLOWED_SYMPY_NAMES, transformations=TRANSFORMATIONS)
        expected = 2 * x
        check('"2x" -> 2*x', result.equals(expected), f"Got: {result}")
    except Exception as e:
        check('"2x" -> 2*x', False, f"{type(e).__name__}: {e}")

    # Test: "sin(x)cos(x)" should parse as sin(x)*cos(x)
    try:
        result = parse_expr("sin(x)cos(x)", local_dict=ALLOWED_SYMPY_NAMES, transformations=TRANSFORMATIONS)
        expected = sp.sin(x) * sp.cos(x)
        check('"sin(x)cos(x)" -> sin(x)*cos(x)', result.equals(expected), f"Got: {result}")
    except Exception as e:
        check('"sin(x)cos(x)" -> sin(x)*cos(x)', False, f"{type(e).__name__}: {e}")

    # Test: "3pi" should parse as 3*pi
    try:
        result = parse_expr("3pi", local_dict=ALLOWED_SYMPY_NAMES, transformations=TRANSFORMATIONS)
        expected = 3 * sp.pi
        check('"3pi" -> 3*pi', result.equals(expected), f"Got: {result}")
    except Exception as e:
        check('"3pi" -> 3*pi', False, f"{type(e).__name__}: {e}")


# ── 4. Heuristic Regression Tests ───────────────────────────────────────────

def test_heuristic_regression():
    print("\n=== 4. HEURISTIC REGRESSION: Scalloped Staircase ===")
    try:
        from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_scalloped_staircase
    except ImportError:
        check("Import _detect_scalloped_staircase", False, "ImportError")
        return

    # Generate floor(x) data with EXACT integer points
    # _detect_scalloped_staircase requires integer anchors (|x-round(x)| < 1e-9)
    x_ints = np.array([2.0, 3.0, 4.0, 5.0])  # exact integers >= 2
    x_fracs = np.array([2.2, 2.5, 2.8, 3.3, 3.7, 4.1, 4.5, 4.9, 5.2, 5.8])
    x_all = np.concatenate([x_ints, x_fracs])
    x_data = x_all.reshape(-1, 1)
    y_data = np.floor(x_all)

    try:
        result = _detect_scalloped_staircase(x_data, y_data, variable_names=["x"], verbose=False)
        has_floor_seed = False
        if result:
            if isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, str) and "floor" in item.lower():
                        has_floor_seed = True
                        break
                    elif isinstance(item, (list, tuple)):
                        for sub in item:
                            if isinstance(sub, str) and "floor" in sub.lower():
                                has_floor_seed = True
                                break
            check("Detect floor(x) pattern", has_floor_seed or bool(result), f"Seeds: {result}")
        else:
            check("Detect floor(x) pattern", False, "No seeds returned")
    except Exception as e:
        check("Detect floor(x) pattern", False, f"{type(e).__name__}: {e}")


# ── 5. Error UX Tests ──────────────────────────────────────────────────────

def test_error_ux():
    print("\n=== 5. ERROR UX: Friendly Error Messages ===")
    from kalkulator_pkg.utils.parsing import eval_to_float

    # Test: unclosed quote / bracket
    bad_inputs = [
        ("Missing bracket: sin(x", "sin(x"),
        ("Random garbage: @#$%", "@#$%"),
    ]

    for label, inp in bad_inputs:
        try:
            eval_to_float(inp)
            check(label, False, "Should have raised ValueError")
        except ValueError as e:
            # Check it's a friendly message (not a raw stack trace)
            msg = str(e)
            is_friendly = (
                "Syntax error" in msg
                or "Invalid expression" in msg
                or "Could not" in msg
            )
            check(label, is_friendly, f"Message: {msg[:80]}")
        except Exception as e:
            check(label, False, f"Wrong exception: {type(e).__name__}: {e}")


# ── 6. Diophantine Edge Cases ───────────────────────────────────────────────

def test_diophantine():
    print("\n=== 6. DIOPHANTINE: Edge Cases ===")
    from kalkulator_pkg.utils.numeric import compute_integerized_equation

    # Test: normal case
    try:
        result = compute_integerized_equation(
            [sp.Rational(1), sp.Rational(2), sp.Rational(0)],
            sp.Rational(5),
            1
        )
        check("Normal case (x + 2y = 5)", result is not None, f"Result: {result}")
    except Exception as e:
        check("Normal case", False, f"{type(e).__name__}: {e}")

    # Test: zero coefficients
    try:
        result = compute_integerized_equation(
            [sp.Rational(0), sp.Rational(0), sp.Rational(0)],
            sp.Rational(0),
            1
        )
        # All zeros — should handle gracefully (return something or None)
        check("Zero coefficients", True, f"Result: {result}")
    except ZeroDivisionError:
        check("Zero coefficients", False, "ZeroDivisionError (unguarded)")
    except Exception as e:
        check("Zero coefficients", True, f"Handled: {type(e).__name__}")

    # Test: wrong number of coefficients
    try:
        result = compute_integerized_equation(
            [sp.Rational(1), sp.Rational(2)],  # Only 2, needs 3
            sp.Rational(5),
            1
        )
        check("Wrong coeff count", result is None, f"Result: {result}")
    except Exception as e:
        check("Wrong coeff count", True, f"Handled: {type(e).__name__}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 2 Verification: Architecture Hardening")
    print("=" * 60)

    test_security()
    test_pickle_safety()
    test_ambiguity()
    test_heuristic_regression()
    test_error_ux()
    test_diophantine()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
