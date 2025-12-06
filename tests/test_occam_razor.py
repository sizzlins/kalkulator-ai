"""
Test Occam's Razor - Simplicity Preference for Function Finder

These tests verify that simple equations are preferred over complex ones
when both fit the data perfectly.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data


def test_linear_offset():
    """h(0)=5, h(1)=7, h(2)=9 should give h(t) = 2*t + 5, NOT a cubic."""
    print("\n=== LINEAR OFFSET ===")
    print("Expected: h(t) = 2*t + 5")

    data = [([0], 5), ([1], 7), ([2], 9)]
    _, result, _, _ = find_function_from_data(data, ["t"])
    print(f"Result: {result}")

    # Should be a simple linear (2*t + 5 or equivalent)
    # Should NOT contain cubic or quadratic terms
    if result:
        result_clean = result.replace(" ", "")
        has_linear = ("2*t" in result_clean or "t" in result_clean) and "5" in result
        no_cubic = "^3" not in result_clean
        no_exp = "exp" not in result_clean
        if has_linear and no_cubic and no_exp:
            print("[PASS] Linear offset")
            return True
    print(f"[FAIL] Linear offset: {result}")
    return False


def test_pure_cosine():
    """w(0)=1, w(1.57)=0, w(3.14)=-1, w(6.28)=1 should give w(t) = cos(t)."""
    print("\n=== PURE COSINE ===")
    print("Expected: w(t) = cos(t)")

    import math

    data = [([0], 1), ([math.pi / 2], 0), ([math.pi], -1), ([2 * math.pi], 1)]
    _, result, _, _ = find_function_from_data(data, ["t"])
    print(f"Result: {result}")

    if result and "cos(t)" in result and "exp" not in result:
        print("[PASS] Pure cosine")
        return True
    print(f"[FAIL] Pure cosine: {result}")
    return False


def test_simple_product():
    """V(2,10)=20, V(5,5)=25, V(0.5,100)=50 should give V(I,R) = I*R."""
    print("\n=== SIMPLE PRODUCT (Ohm's Law) ===")
    print("Expected: V(I,R) = I*R")

    data = [([2, 10], 20), ([5, 5], 25), ([0.5, 100], 50)]
    _, result, _, _ = find_function_from_data(data, ["I", "R"])
    print(f"Result: {result}")

    if result:
        result_clean = result.replace(" ", "")
        if ("I*R" in result_clean or "R*I" in result_clean) and "^" not in result_clean:
            print("[PASS] Simple product")
            return True
    print(f"[FAIL] Simple product: {result}")
    return False


def test_triple_product():
    """U(1,10,1)=10, U(2,9.8,5)=98, U(5,2,2)=20 should give U(m,g,h) = m*g*h."""
    print("\n=== TRIPLE PRODUCT (Potential Energy) ===")
    print("Expected: U(m,g,h) = m*g*h")

    data = [([1, 10, 1], 10), ([2, 9.8, 5], 98), ([5, 2, 2], 20)]
    _, result, _, _ = find_function_from_data(data, ["m", "g", "h"])
    print(f"Result: {result}")

    if result:
        result_clean = result.replace(" ", "")
        # Check for m*g*h in any order
        if (
            "m*g*h" in result_clean
            or "m*h*g" in result_clean
            or "g*m*h" in result_clean
            or "g*h*m" in result_clean
            or "h*m*g" in result_clean
            or "h*g*m" in result_clean
        ) and "^" not in result_clean:
            print("[PASS] Triple product")
            return True
    print(f"[FAIL] Triple product: {result}")
    return False


def test_newtons_law():
    """F(10,2)=20, F(5,5)=25, F(100,0.1)=10 should give F(m,a) = m*a."""
    print("\n=== NEWTON'S LAW ===")
    print("Expected: F(m,a) = m*a")

    data = [([10, 2], 20), ([5, 5], 25), ([100, 0.1], 10)]
    _, result, _, _ = find_function_from_data(data, ["m", "a"])
    print(f"Result: {result}")

    if result:
        result_clean = result.replace(" ", "")
        if (
            ("m*a" in result_clean or "a*m" in result_clean)
            and "sin" not in result_clean
            and "exp" not in result_clean
        ):
            print("[PASS] Newton's law")
            return True
    print(f"[FAIL] Newton's law: {result}")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("OCCAM'S RAZOR TEST SUITE")
    print("=" * 60)

    results = []
    results.append(("LinearOffset", test_linear_offset()))
    results.append(("PureCosine", test_pure_cosine()))
    results.append(("SimpleProduct", test_simple_product()))
    results.append(("TripleProduct", test_triple_product()))
    results.append(("NewtonsLaw", test_newtons_law()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        status = "[PASS]" if r else "[FAIL]"
        print(f"  {name}: {status}")
    print(f"\nTotal: {passed}/{total}")
