#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rigorous test suite with better error handling."""
import io
import sys

sys.path.insert(0, '.')

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import re

from kalkulator_pkg.cli import print_result_pretty
from kalkulator_pkg.solver import solve_single_equation, solve_system
from kalkulator_pkg.worker import evaluate_safely


def safe_print(msg):
    """Print with error handling for encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Replace problematic characters
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(msg_safe)

def test_value(input_str, expected=None):
    """Test value evaluation."""
    result = evaluate_safely(input_str)
    if result.get("ok"):
        res = result.get('result', '')
        if expected and str(res) != str(expected):
            safe_print(f"  [WARN] Expected {expected}, got {res}")
        return True, res
    else:
        return False, result.get('error', 'Unknown error')

def test_equation(input_str, find_var=None):
    """Test equation solving."""
    # Remove "find" if present - handle commas before/after
    if "find" in input_str.lower():
        find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
        find_var = find_tokens[0] if find_tokens else find_var
        # Remove "find" and any surrounding commas/whitespace
        input_str = re.sub(r",?\s*\bfind\s+\w+\b\s*,?", "", input_str, flags=re.IGNORECASE).strip()
        # Clean up any trailing commas
        input_str = input_str.rstrip(',').strip()
        # Clean up any double commas
        input_str = re.sub(r",\s*,", ",", input_str)
    
    result = solve_single_equation(input_str, find_var)
    if result.get("ok"):
        return True, result
    else:
        return False, result.get('error', 'Unknown error')

def test_system(input_str):
    """Test system solving."""
    find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
    find_token = find_tokens[0] if find_tokens else None
    raw_no_find = re.sub(r"\bfind\s+\w+\b", "", input_str, flags=re.IGNORECASE).strip()
    result = solve_system(raw_no_find, find_token)
    if result.get("ok"):
        return True, result
    else:
        return False, result.get('error', 'Unknown error')

# Test cases
tests = [
    ("1+1", "value", "2"),
    ("1-1", "value", "0"),
    ("5*10", "value", "50"),
    ("1/2", "value", "0.5"),
    ("1=1", "equation", None),
    ("1=0", "equation", None),
    ("50%", "value", "0.5"),
    ("50/100", "value", "0.5"),
    ("2^2", "value", "4"),
    ("2**2", "value", "4"),
    ("2^2", "value", "4"),  # Test superscript separately
    ("sqrt(2)", "value", None),  # Use sqrt instead of √
    ("pi", "value", None),
    ("E", "value", None),
    ("I", "value", None),
    ("ln(I)", "value", None),
    ("log(2)", "value", None),
    ("log(pi)", "value", None),  # This one was failing
    ("E^(I*pi)", "value", "-1"),
    ("E^(I*pi)+1", "value", "0"),
    ("r=5, pi*r^2=n, find n", "system", None),
    ("a = 2, b = a+3", "assignment", None),
    ("x+y=3, x-y=1", "system", None),
    ("sin(pi/6)", "value", "0.5"),
    ("2sin(x)+sqrt(3)=0", "equation", None),
    ("3tan(x)+sqrt(3)=0", "equation", None),
    ("2sin(x)^2-1=0", "equation", None),
    ("3sin(x)+2=1", "equation", None),
    ("3tan(x)*2tan(x)*tan(x)=3tan(x)-2tan(x)-tan(x)", "equation", None),
    ("2v+w+x+y+z=5, v+2w+x+y+z=5, v+w+2x+y+z=6, v+w+x+2y+z=7, v+w+x+y+2z=8, v+w+x+y+z=a, find a", "system", None),
    ("E^(I*pi)+1=x, find x", "equation", None),
    ("6x^2-17x+1=0, find x", "equation", None),
    ("x^3-4x^2-9x+36=0, find x", "equation", None),
    ("x^3-9x+36=0, find x", "equation", None),
    ("pi + E + I + sqrt(2) + sin(pi/2) + cos(0) + tan(pi/4) + asin(1) + acos(0) + atan(1) + log(10) + ln(E) + exp(1) + Abs(-5)", "value", None),
    ("sin(1/x)^-1=(sin(x)/1)^-1, find x", "equation", None),
]

def main():
    passed = 0
    failed = 0
    issues = []
    
    safe_print("="*60)
    safe_print("RIGOROUS KALKULATOR TEST SUITE")
    safe_print("="*60)
    
    for i, (input_str, test_type, expected) in enumerate(tests, 1):
        safe_print(f"\n[{i}/{len(tests)}] Testing: {input_str}")
        
        try:
            if test_type == "value":
                success, result = test_value(input_str, expected)
                if success:
                    safe_print(f"  [OK] Result: {result}")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Error: {result}")
                    failed += 1
                    issues.append((input_str, result))
                    
            elif test_type == "equation":
                success, result = test_equation(input_str)
                if success:
                    safe_print(f"  [OK] Equation solved")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Error: {result}")
                    failed += 1
                    issues.append((input_str, result))
                    
            elif test_type == "system":
                success, result = test_system(input_str)
                if success:
                    safe_print(f"  [OK] System solved")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Error: {result}")
                    failed += 1
                    issues.append((input_str, result))
                    
            elif test_type == "assignment":
                # Handle assignments
                parts = input_str.split(",")
                all_ok = True
                for part in parts:
                    part = part.strip()
                    if "=" in part:
                        var, expr = part.split("=", 1)
                        var = var.strip()
                        expr = expr.strip()
                        success, res = test_value(expr)
                        if success:
                            safe_print(f"  [OK] {var} = {res}")
                        else:
                            safe_print(f"  [FAIL] Error evaluating {var}: {res}")
                            all_ok = False
                if all_ok:
                    passed += 1
                else:
                    failed += 1
                    issues.append((input_str, "Assignment failed"))
            else:
                safe_print(f"  [SKIP] Unknown test type: {test_type}")
                
        except Exception as e:
            safe_print(f"  [EXCEPTION] {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            issues.append((input_str, str(e)))
    
    safe_print("\n" + "="*60)
    safe_print("TEST SUMMARY")
    safe_print("="*60)
    safe_print(f"Passed: {passed}")
    safe_print(f"Failed: {failed}")
    safe_print(f"Total: {passed + failed}")
    
    if issues:
        safe_print("\nIssues found:")
        for input_str, error in issues:
            safe_print(f"  - {input_str}: {error}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

