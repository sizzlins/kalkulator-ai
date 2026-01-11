
import sympy as sp
import sys

def mock_simplify(expr_str):
    print(f"Original: {expr_str}")
    try:
        # 1. Parse
        # simplistic parsing for the mock, real code uses more context
        # but for exp(log(x**x)) it should work directly
        expr = sp.sympify(expr_str)
        print(f"Parsed:   {expr}")
        
        # 2. Simplify
        simplified = sp.simplify(expr)
        print(f"Simplified: {simplified}")
        
        return str(simplified)
    except Exception as e:
        print(f"Error: {e}")
        return expr_str

def verify():
    print("--- Verify Simplification ---")
    
    cases = [
        "exp(log(x**x))",
        "1/(1/x)",
        "x + 0",
        "exp(log(x))",
        "log(exp(x))"
    ]
    
    passed = 0
    for case in cases:
        print(f"\nTesting: {case}")
        res = mock_simplify(case)
        
        # Check expectations
        if case == "exp(log(x**x))" and res == "x**x":
            print("PASS")
            passed += 1
        elif case == "1/(1/x)" and res == "x":
            print("PASS")
            passed += 1
        elif case == "x + 0" and res == "x":
             print("PASS")
             passed += 1
        elif case == "exp(log(x))" and res == "x":
             print("PASS")
             passed += 1
        elif case == "log(exp(x))" and res == "x":
             print("PASS")
             passed += 1
        else:
             print(f"FAIL: Expected simplified form, got {res}")

    if passed == len(cases):
        print("\nALL TESTS PASSED")
    else:
        print(f"\n{len(cases) - passed} TESTS FAILED")

if __name__ == "__main__":
    verify()
