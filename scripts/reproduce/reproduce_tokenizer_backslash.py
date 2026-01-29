
import sys
import os
import tokenize
import io

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.tokenizer import transform_input

def test_input(val, label):
    print(f"\n--- Testing {label} ---")
    print(f"Input: '{val}'")
    try:
        result = transform_input(val)
        print(f"Success! Result: '{result}'")
    except Exception as e:
        print(f"FAIL: {e}")

# Test trailing backslash
test_input("f(x)=4.9x^2\\", "Trailing Backslash")

# Test backslash elsewhere
test_input("f(x)=4.9x^2 \\ ", "Backslash with space")
