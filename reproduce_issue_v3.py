from kalkulator_pkg.parser import preprocess, parse_preprocessed
from kalkulator_pkg.types import ValidationError
import sys

def test_fixes():
    print("--- Verifying Fixes ---")
    
    # Test 1: f(1) undefined function
    expr = "f(1)"
    print(f"\nExpression: '{expr}'")
    try:
        pre = preprocess(expr)
        print(f"Preprocessed: '{pre}'")
        # parsed = parse_preprocessed(pre) # parsing handled within preprocess? No.
        # Check if validation error raised during preprocess (parser logic I added triggers in preprocess?)
        # My added logic was in verify_undefined_functions inside preprocess?
        # Let's check parser.py again. I added it to preprocess.
    except ValidationError as e:
        print(f"SUCCESS: Caught expected ValidationError: {e}")
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {type(e).__name__}: {e}")
    else:
         print(f"FAILURE: No exception raised for undefined function '{expr}'. Check implementation.")

    # Test 2: f(x)= handled by app.py logic (trailing =), hard to unit test here without app context.
    # But we can test parser response if we passed it (it should fail syntax if passed to parser)
    
    print("\n--- Done ---")

if __name__ == "__main__":
    test_fixes()
