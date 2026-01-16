
from kalkulator_pkg.parser import preprocess_expression
from kalkulator_pkg.types import ValidationError
import sys

def test(case, expected):
    print(f"Testing: '{case}'")
    try:
        result = preprocess_expression(case, skip_exponent_conversion=False)
        print(f"  Result: '{result}'")
        
        if expected == "ERROR":
            print("  FAILED: Expected Error, got success")
            return False
            
        # Normalize result for comparison (remove spaces)
        norm_res = result.replace(" ", "")
        norm_exp = expected.replace(" ", "")
        
        if norm_res == norm_exp:
            print("  PASS")
            return True
        else:
            print(f"  FAILED: Expected '{expected}', got '{result}'")
            return False
            
    except ValidationError as e:
        print(f"  Caught Expected Error: {e}")
        if expected == "ERROR":
            print("  PASS")
            return True
        else:
            print(f"  FAILED: Unexpected Error")
            return False
    except Exception as e:
        print(f"  CRASH: {e}")
        return False

def verify():
    tests = [
        # 1. Implicit Mult
        ("2x", "2*x"),
        ("x y", "x*y"),
        ("2(x)", "2*(x)"),
        ("(x)y", "(x)*y"),
        ("x(y)", "x(y)"), # Function call? Or x*y? My logic assumes Call.
        
        # 2. Syntax Sugar
        ("x^2", "x**2"),
        ("x**2", "x**2"),
        
        # 3. Functions
        ("sin(x)", "sin(x)"),
        ("diff(x, x)", "diff(x, x)"), # Comma check
        
        # 4. Safety
        ("import os", "ERROR"),
        ("__class__", "ERROR"),
        
        # 5. Unicode
        ("√x", "sqrt(x)"),
        ("π", "pi"),
        ("2π", "2*pi"), # Implicit mult with unicode
    ]
    
    failures = 0
    for inp, exp in tests:
        if not test(inp, exp):
            failures += 1
            
    if failures == 0:
        print("\nALL TESTS PASSED")
    else:
        print(f"\n{failures} TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    verify()
