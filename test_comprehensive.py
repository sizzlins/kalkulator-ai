#!/usr/bin/env python
"""Comprehensive test suite for Kalkulator."""
import sys

sys.path.insert(0, '.')

import json

from kalkulator_pkg.cli import print_result_pretty
from kalkulator_pkg.solver import solve_single_equation, solve_system
from kalkulator_pkg.worker import evaluate_safely

# Test cases: (input, expected_type, description)
test_cases = [
    # Basic arithmetic
    ("1+1", "value", "Addition"),
    ("1-1", "value", "Subtraction"),
    ("5*10", "value", "Multiplication"),
    ("1/2", "value", "Division"),
    ("1=1", "equation", "True equation"),
    ("1=0", "equation", "False equation"),
    
    # Percentages and fractions
    ("50%", "value", "Percentage"),
    ("50/100", "value", "Fraction"),
    
    # Exponents
    ("2^2", "value", "Power with ^"),
    ("2**2", "value", "Power with **"),
    ("2²", "value", "Power with superscript"),
    
    # Roots and constants
    ("√(2)", "value", "Square root"),
    ("pi", "value", "Pi constant"),
    ("E", "value", "Euler's number"),
    ("I", "value", "Imaginary unit"),
    
    # Logarithms
    ("ln(I)", "value", "Natural log of I"),
    ("log(2)", "value", "Log base 10"),
    ("log(pi)", "value", "Log of pi"),
    
    # Complex expressions
    ("E^(I*pi)", "value", "Euler's identity"),
    ("E^(I*pi)+1", "value", "Euler's identity + 1"),
    
    # System solving
    ("r=5, pi*r^2=n, find n", "system", "Circle area calculation"),
    ("a = 2, b = a+3", "assignment", "Variable assignment"),
    ("x+y=3, x-y=1", "system", "Linear system"),
    
    # Trigonometry
    ("sin(pi/6)", "value", "Sine"),
    ("2sin(x)+sqrt(3)=0", "equation", "Trigonometric equation"),
    ("3tan(x)+sqrt(3)=0", "equation", "Tangent equation"),
    ("2sin(x)^2-1=0", "equation", "Sine squared equation"),
    ("3sin(x)+2=1", "equation", "Sine equation"),
    ("3tan(x)*2tan(x)*tan(x)=3tan(x)-2tan(x)-tan(x)", "equation", "Complex trig equation"),
    
    # Complex system
    ("2v+w+x+y+z=5, v+2w+x+y+z=5, v+w+2x+y+z=6, v+w+x+2y+z=7, v+w+x+y+2z=8, v+w+x+y+z=a, find a", "system", "5-variable system"),
    
    # Complex equation solving
    ("E^(I*pi)+1=x, find x", "equation", "Euler's identity equation"),
    ("6x^2-17x+1=0, find x", "equation", "Quadratic equation"),
    ("x^3-4x^2-9x+36=0, find x", "equation", "Cubic equation"),
    ("x^3-9x+36=0, find x", "equation", "Cubic equation 2"),
    
    # Complex expression
    ("pi + E + I + sqrt(2) + sin(pi/2) + cos(0) + tan(pi/4) + asin(1) + acos(0) + atan(1) + log(10) + ln(E) + exp(1) + Abs(-5)", "value", "Complex expression"),
    
    # Tricky equation
    ("sin(1/x)^-1=(sin(x)/1)^-1, find x", "equation", "Complex trigonometric equation"),
]

def test_input(input_str, test_type, description):
    """Test a single input."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Input: {input_str}")
    print(f"Type: {test_type}")
    print(f"{'='*60}")
    
    try:
        if test_type == "value":
            result = evaluate_safely(input_str)
            if result.get("ok"):
                print(f"[OK] Result: {result.get('result')}")
                if result.get('approx'):
                    print(f"  Approx: {result.get('approx')}")
            else:
                print(f"[FAIL] Error: {result.get('error')}")
                return False
                
        elif test_type == "equation":
            # Extract find token if present
            import re
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
            find_token = find_tokens[0] if find_tokens else None
            # Remove "find" from input string and clean up trailing commas
            equation_str = re.sub(
                r",?\s*\bfind\s+\w+\b\s*,?", "", input_str, flags=re.IGNORECASE
            ).strip().rstrip(',').strip()
            result = solve_single_equation(equation_str, find_token)
            if result.get("ok"):
                print(f"[OK] Success")
                print_result_pretty(result, "human")
            else:
                print(f"[FAIL] Error: {result.get('error')}")
                return False
                
        elif test_type == "system":
            # Extract find token if present
            import re
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
            find_token = find_tokens[0] if find_tokens else None
            raw_no_find = re.sub(
                r"\bfind\s+\w+\b", "", input_str, flags=re.IGNORECASE
            ).strip()
            result = solve_system(raw_no_find, find_token)
            if result.get("ok"):
                print(f"[OK] Success")
                print_result_pretty(result, "human")
            else:
                print(f"[FAIL] Error: {result.get('error')}")
                return False
                
        elif test_type == "assignment":
            # This should be handled by REPL, but we can test parsing
            parts = input_str.split(",")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    var, expr = part.split("=", 1)
                    var = var.strip()
                    expr = expr.strip()
                    result = evaluate_safely(expr)
                    if result.get("ok"):
                        print(f"[OK] {var} = {result.get('result')}")
                    else:
                        print(f"[FAIL] Error evaluating {var}: {result.get('error')}")
                        return False
        else:
            print(f"[FAIL] Unknown test type: {test_type}")
            return False
            
        return True
        
    except Exception as e:
        print(f"[FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("COMPREHENSIVE KALKULATOR TEST SUITE")
    print("="*60)
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for input_str, test_type, description in test_cases:
        try:
            success = test_input(input_str, test_type, description)
            if success:
                passed += 1
            else:
                failed += 1
                failed_tests.append((input_str, description))
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n[FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_tests.append((input_str, description))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed_tests:
        print("\nFailed tests:")
        for input_str, description in failed_tests:
            print(f"  - {description}: {input_str}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

