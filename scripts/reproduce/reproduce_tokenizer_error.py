
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.tokenizer import transform_input

try:
    text = "f(x)=4.9x^2"
    print(f"Testing input: '{text}'")
    result = transform_input(text)
    print(f"Success! Result: '{result}'")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

# Also test with what might be coming from REPL if it preprocesses differently
# e.g. "f(x) = 4.9x^2" or similar
try:
    text = "f(x) = 4.9x^2" 
    print(f"Testing input with spaces: '{text}'")
    result = transform_input(text)
    print(f"Success! Result: '{result}'")
except Exception as e:
    print(f"FAIL: {e}")
