#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Edge case and variation tests."""
import io
import sys

sys.path.insert(0, '.')

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import re

from kalkulator_pkg.solver import solve_single_equation, solve_system
from kalkulator_pkg.worker import evaluate_safely


def safe_print(msg):
    """Print with error handling."""
    try:
        print(msg)
    except UnicodeEncodeError:
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(msg_safe)

def test_value(input_str):
    """Test value evaluation."""
    result = evaluate_safely(input_str)
    if result.get("ok"):
        return True, result.get('result', '')
    else:
        return False, result.get('error', 'Unknown error')

def test_equation(input_str, find_var=None):
    """Test equation solving."""
    if "find" in input_str.lower():
        find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
        find_var = find_tokens[0] if find_tokens else find_var
        input_str = re.sub(r",?\s*\bfind\s+\w+\b\s*,?", "", input_str, flags=re.IGNORECASE).strip()
        input_str = input_str.rstrip(',').strip()
    
    result = solve_single_equation(input_str, find_var)
    if result.get("ok"):
        return True, result
    else:
        return False, result.get('error', 'Unknown error')

# Edge cases and variations
edge_cases = [
    # Arithmetic edge cases
    ("0+0", "value"),
    ("0*0", "value"),
    ("1/0", "value"),  # Should handle division by zero
    ("0/1", "value"),
    ("0^0", "value"),  # 0^0 is undefined
    ("1^0", "value"),
    ("0^1", "value"),
    
    # Large numbers
    ("999999999999999999", "value"),
    ("1e10", "value"),
    ("1e-10", "value"),
    
    # Negative numbers
    ("-1", "value"),
    ("-1+1", "value"),
    ("-1*-1", "value"),
    ("-1^2", "value"),
    ("(-1)^2", "value"),
    
    # Parentheses variations
    ("(1+1)", "value"),
    ("((1+1))", "value"),
    ("((1+1)+1)", "value"),
    ("1+(1+1)", "value"),
    
    # Implicit multiplication
    ("2x", "value"),  # Should error or treat as 2*x with x undefined
    ("2(x+1)", "value"),
    ("(x+1)2", "value"),
    
    # Trigonometric functions
    ("sin(0)", "value"),
    ("cos(0)", "value"),
    ("tan(0)", "value"),
    ("sin(pi)", "value"),
    ("cos(pi)", "value"),
    ("tan(pi)", "value"),
    ("sin(pi/2)", "value"),
    ("cos(pi/2)", "value"),
    ("asin(0)", "value"),
    ("acos(1)", "value"),
    ("atan(0)", "value"),
    ("asin(1)", "value"),
    ("acos(0)", "value"),
    ("atan(1)", "value"),
    
    # Logarithmic functions
    ("ln(1)", "value"),
    ("ln(E)", "value"),
    ("log(1)", "value"),
    ("log(10)", "value"),
    ("ln(0)", "value"),  # Should handle -infinity
    ("log(0)", "value"),
    
    # Exponential
    ("exp(0)", "value"),
    ("exp(1)", "value"),
    ("E^0", "value"),
    ("E^1", "value"),
    
    # Complex numbers
    ("I^2", "value"),
    ("I^3", "value"),
    ("I^4", "value"),
    ("1+I", "value"),
    ("1-I", "value"),
    ("I*I", "value"),
    
    # Square roots
    ("sqrt(0)", "value"),
    ("sqrt(1)", "value"),
    ("sqrt(4)", "value"),
    ("sqrt(-1)", "value"),  # Should be I
    
    # Absolute value
    ("Abs(0)", "value"),
    ("Abs(1)", "value"),
    ("Abs(-1)", "value"),
    ("Abs(I)", "value"),
    
    # Percentage variations
    ("0%", "value"),
    ("100%", "value"),
    ("200%", "value"),
    ("-50%", "value"),
    
    # Equations with edge cases
    ("x=0", "equation"),
    ("0=x", "equation"),
    ("x+x=0", "equation"),
    ("x-x=0", "equation"),
    ("x*x=0", "equation"),
    ("x/x=1", "equation"),  # Should handle x != 0
    ("x^2=0", "equation"),
    ("x^2=1", "equation"),
    ("x^2=-1", "equation"),
    
    # System edge cases
    ("x=1, y=2", "system"),
    ("x=0, y=0", "system"),
    ("x+y=0, x-y=0", "system"),
    ("x=1, x=2", "system_no_solution"),  # Contradictory - expected no solution
    ("x+y=1, x+y=2", "system_no_solution"),  # Contradictory - expected no solution
    
    # Trigonometric equations
    ("sin(x)=0", "equation"),
    ("cos(x)=0", "equation"),
    ("tan(x)=0", "equation"),
    ("sin(x)=1", "equation"),
    ("cos(x)=1", "equation"),
    
    # Polynomial equations
    ("x=0", "equation"),
    ("x-1=0", "equation"),
    ("x^2-1=0", "equation"),
    ("x^2-4=0", "equation"),
    ("x^3-8=0", "equation"),
    ("x^4-16=0", "equation"),
]

def main():
    passed = 0
    failed = 0
    issues = []
    
    safe_print("="*60)
    safe_print("EDGE CASE TEST SUITE")
    safe_print("="*60)
    
    for i, (input_str, test_type) in enumerate(edge_cases, 1):
        safe_print(f"\n[{i}/{len(edge_cases)}] Testing: {input_str}")
        
        try:
            if test_type == "value":
                success, result = test_value(input_str)
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
                    safe_print(f"  [OK] Solved")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Error: {result}")
                    failed += 1
                    issues.append((input_str, result))
                    
            elif test_type == "system":
                # Handle system
                find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
                find_token = find_tokens[0] if find_tokens else None
                raw_no_find = re.sub(r"\bfind\s+\w+\b", "", input_str, flags=re.IGNORECASE).strip()
                result = solve_system(raw_no_find, find_token)
                if result.get("ok"):
                    safe_print(f"  [OK] Solved")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Error: {result.get('error')}")
                    failed += 1
                    issues.append((input_str, result.get('error')))
                    
            elif test_type == "system_no_solution":
                # Handle contradictory systems - expected to have no solution
                find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
                find_token = find_tokens[0] if find_tokens else None
                raw_no_find = re.sub(r"\bfind\s+\w+\b", "", input_str, flags=re.IGNORECASE).strip()
                result = solve_system(raw_no_find, find_token)
                if not result.get("ok") or "no solution" in str(result.get("error", "")).lower():
                    safe_print(f"  [OK] Correctly reports no solution (contradictory)")
                    passed += 1
                else:
                    safe_print(f"  [FAIL] Should have no solution but got: {result}")
                    failed += 1
                    issues.append((input_str, "Expected no solution"))
                    
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
        for input_str, error in issues[:10]:  # Show first 10
            safe_print(f"  - {input_str}: {error}")
        if len(issues) > 10:
            safe_print(f"  ... and {len(issues) - 10} more")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

