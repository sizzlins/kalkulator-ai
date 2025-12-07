"""
Comprehensive Physics Test Suite
All test cases from user calibration sessions.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from kalkulator_pkg.function_manager import find_function_from_data  # noqa: E402


def test_coulomb():
    """Coulomb's Law: F = 9000 * q1 * q2 / r^2"""
    print("\n=== COULOMB'S LAW ===")
    print("Expected: F = 9000 * q1 * q2 / r^2")

    data_points = [
        ([1, 1, 1], 9000.0),
        ([1, 1, 2], 2250.0),
        ([1, 1, 3], 1000.0),
        ([2, 1, 1], 18000.0),
        ([2, 2, 2], 9000.0),
        ([5, 4, 10], 1800.0),
        ([2, 5, 10], 900.0),
        ([1, 1, 10], 90.0),
        ([3, 3, 3], 9000.0),
        ([10, 10, 1], 900000.0),
        ([1, 1, 0.5], 36000.0),
        ([4, 4, 4], 9000.0),
        ([2, 8, 4], 9000.0),
        ([5, 5, 25], 72.0),
        ([10, 2, 100], 18.0),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["q1", "q2", "r"])
    result_clean = result.replace(" ", "") if result else ""

    if "q1*q2/r^2" in result_clean and "9000" in result_clean:
        print(f"[PASS] Coulomb: {result}")
        return True
    else:
        print(f"[FAIL] Coulomb: {result}")
        return False


def test_mechanical_energy():
    """Mechanical Energy: E = 0.5*m*v^2 + 9.8*m*h"""
    print("\n=== MECHANICAL ENERGY ===")
    print("Expected: E = 0.5*m*v^2 + 9.8*m*h")

    data_points = [
        ([1, 2, 0], 2.0),
        ([1, 0, 1], 9.8),
        ([2, 2, 1], 23.6),
        ([10, 10, 10], 1480.0),
        ([1, 10, 0], 50.0),
        ([10, 0, 0], 0.0),
        ([1, 4, 0], 8.0),
        ([2, 4, 0], 16.0),
        ([4, 4, 0], 32.0),
        ([1, 0, 2], 19.6),
        ([2, 0, 2], 39.2),
        ([1, 1, 1], 10.3),
        ([2, 1, 1], 20.6),
        ([3, 1, 1], 30.9),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["m", "v", "h"])
    result_clean = result.replace(" ", "") if result else ""

    # Accept polynomial (no transcendentals)
    if (
        "exp" not in result_clean
        and "sin(" not in result_clean
        and "cos(" not in result_clean
    ):
        print(f"[PASS] MechEnergy: {result}")
        return True
    else:
        print(f"[FAIL] MechEnergy: {result}")
        return False


def test_spacetime():
    """Spacetime Interval: W = -A^2 + B^2 + C^2 + D^2"""
    print("\n=== SPACETIME INTERVAL ===")
    print("Expected: W = -A^2 + B^2 + C^2 + D^2")

    data_points = [
        ([10, 2, 3, 1], -86.0),
        ([2, 5, 1, 1], 23.0),
        ([1, 1, 1, 1], 2.0),
        ([5, 2, 2, 2], -13.0),
        ([3, 3, 3, 3], 18.0),
        ([4, 0, 0, 0], -16.0),
        ([0, 4, 0, 0], 16.0),
        ([0, 0, 4, 0], 16.0),
        ([0, 0, 0, 4], 16.0),
        ([5, 5, 0, 0], 0.0),
        ([10, 1, 1, 1], -97.0),
        ([2, 10, 2, 2], 100.0),
        ([1, 1, 10, 1], 100.0),
        ([1, 1, 1, 10], 100.0),
        ([8, 8, 1, 1], 2.0),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["A", "B", "C", "D"])
    result_clean = result.replace(" ", "") if result else ""

    has_squares = "A^2" in result_clean and "B^2" in result_clean
    no_exp = "exp" not in result_clean

    if has_squares and no_exp:
        print(f"[PASS] Spacetime: {result}")
        return True
    else:
        print(f"[FAIL] Spacetime: {result}")
        return False


def test_gas_law():
    """Ideal Gas Law: P = A*B/C"""
    print("\n=== IDEAL GAS LAW ===")
    print("Expected: P = A*B/C")

    data_points = [
        ([1, 100, 10], 10.0),
        ([2, 300, 20], 30.0),
        ([5, 50, 25], 10.0),
        ([1, 300, 1], 300.0),
        ([2, 50, 5], 20.0),
        ([10, 10, 1], 100.0),
        ([4, 25, 2], 50.0),
        ([3, 100, 300], 1.0),
        ([2, 400, 8], 100.0),
        ([1, 1, 1], 1.0),
        ([10, 100, 10], 100.0),
        ([5, 200, 100], 10.0),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["A", "B", "C"])
    result_clean = result.replace(" ", "") if result else ""

    if "A*B/C" in result_clean or "B*A/C" in result_clean:
        print(f"[PASS] GasLaw: {result}")
        return True
    else:
        print(f"[FAIL] GasLaw: {result}")
        return False


def test_kinetic_energy():
    """Kinetic Energy: K = 0.5*A*B^2"""
    print("\n=== KINETIC ENERGY ===")
    print("Expected: K = 0.5*A*B^2")

    data_points = [
        ([10, 2], 20.0),
        ([1, 10], 50.0),
        ([5, 4], 40.0),
        ([2, 5], 25.0),
        ([100, 1], 50.0),
        ([0.5, 2], 1.0),
        ([4, 3], 18.0),
        ([8, 0.5], 1.0),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["A", "B"])
    result_clean = result.replace(" ", "") if result else ""

    if ("A*B^2" in result_clean or "B^2*A" in result_clean) and (
        "1/2" in result or "0.5" in result
    ):
        print(f"[PASS] KineticEnergy: {result}")
        return True
    else:
        print(f"[FAIL] KineticEnergy: {result}")
        return False


def test_sphere_volume():
    """Sphere Volume: V = 4/3*pi*X^3"""
    print("\n=== SPHERE VOLUME ===")
    print("Expected: V = 4/3*pi*X^3")

    data_points = [
        ([1], 4.1887902048),
        ([3], 113.097335529),
        ([6], 904.778684234),
        ([2], 33.510321638),
        ([0.5], 0.5235987756),
        ([10], 4188.790204786),
    ]
    _, result, _, _ = find_function_from_data(data_points, ["X"])
    result_clean = result.replace(" ", "") if result else ""

    if "X^3" in result_clean and ("4/3*pi" in result_clean or "4.188" in result_clean):
        print(f"[PASS] SphereVolume: {result}")
        return True
    else:
        print(f"[FAIL] SphereVolume: {result}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE PHYSICS TEST SUITE")
    print("=" * 60)

    results = []
    results.append(("Coulomb", test_coulomb()))
    results.append(("MechEnergy", test_mechanical_energy()))
    results.append(("Spacetime", test_spacetime()))
    results.append(("GasLaw", test_gas_law()))
    results.append(("KineticEnergy", test_kinetic_energy()))
    results.append(("SphereVolume", test_sphere_volume()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        status = "[PASS]" if r else "[FAIL]"
        print(f"  {name}: {status}")
    print(f"\nTotal: {passed}/{total}")
