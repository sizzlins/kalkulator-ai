from kalkulator_pkg.parser import preprocess, parse_preprocessed
import sympy as sp

def test_parsing():
    print("--- Testing Parsing ---")
    
    # Test 1: f(1)
    expr = "f(1)"
    print(f"\nExpression: '{expr}'")
    try:
        pre = preprocess(expr)
        print(f"Preprocessed: '{pre}'")
        parsed = parse_preprocessed(pre)
        print(f"Parsed: {parsed} (Type: {type(parsed)})")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: f(x)=
    expr = "f(x)="
    print(f"\nExpression: '{expr}'")
    try:
        pre = preprocess(expr) # might fail
        print(f"Preprocessed: '{pre}'")
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        
    # Test 3: Formatting check (logic is in app.py, so simple unit test here is hard, skipping)

if __name__ == "__main__":
    test_parsing()
