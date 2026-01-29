
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
        # Mimic parser.py behavior: replace ^ with ** first
        pre_processed = val.replace("^", "**")
        print(f"Pre-processed (parser.py logic): '{pre_processed}'")
        
        # Now call transform_input
        result = transform_input(pre_processed)
        print(f"Success! Result: '{result}'")
    except Exception as e:
        print(f"FAIL: {e}")
        # import traceback
        # traceback.print_exc()

# 1. Exact string from error report?
test_input("f(x)=4.9x^2", "Original Input")

# 2. Maybe there's a problem with implicit multiplication handling in transform_input 
# being combined with tokenizer?
# x**2 -> x ** 2. 
# 4.9x -> 4.9 * x
# Combined: 4.9 * x ** 2

# 3. What if there are invisible characters?
# The user copy-pasted logging output in the prompt.
# But the ERROR happened on typed input.

# 4. Direct checks on tokenize behavior with similar strings
try:
    print("\n--- Direct Tokenize Check ---")
    s = "f(x)=4.9x**2"
    print(f"Tokenizing: '{s}'")
    tokens = list(tokenize.tokenize(io.BytesIO(s.encode('utf-8')).readline))
    for t in tokens:
        print(t)
except Exception as e:
    print(f"Direct Tokenize FAIL: {e}")

