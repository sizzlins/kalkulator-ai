
from kalkulator_pkg.parser import parse_preprocessed, preprocess
import sympy as sp

def test_min_parse():
    # The string reported by user seems to be the result of a Rational output
    expr_str = "min(7106521/791680, 2*x)"
    print(f"Testing expression: '{expr_str}'")
    
    # 1. Check what preprocess does to it
    try:
        pre = preprocess(expr_str)
        print(f"Preprocessed: '{pre}'")
    except Exception as e:
        print(f"Preprocess Error: {e}")
        return

    # 2. Try to parse it
    try:
        res = parse_preprocessed(pre)
        print(f"Parsed Result: {res}")
        print(f"Type: {type(res)}")
    except Exception as e:
        print(f"Parse Error: {e}")
        # If it matches user error: unsupported operand type(s) for /: 'Integer' and 'tuple'
        if "unsupported operand type" in str(e) and "tuple" in str(e):
            print("REPRODUCED: The specific error was caught.")

if __name__ == "__main__":
    test_min_parse()
