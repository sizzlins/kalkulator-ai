
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re

# Mock ALLOWED_SYMPY_NAMES
ALLOWED_NAMES = {
    'tan': sp.tan,
    'sin': sp.sin,
    'cos': sp.cos,
    'x': sp.Symbol('x')
}

def test_parse():
    text = "sin(cos(tan))"
    print(f"Parsing: {text}")
    try:
        # Simulate local dict
        res = parse_expr(text, local_dict=ALLOWED_NAMES)
        print("Parsed Result:", res)
    except Exception as e:
        print(f"Caught Expected Error: {e}")
        check_heuristic(text)

def check_heuristic(text):
    print("\nChecking heuristic:")
    found_issue = False
    for name, obj in ALLOWED_NAMES.items():
        # Check if it's a Function class (not instance or symbol)
        is_func_class = isinstance(obj, type) and issubclass(obj, sp.Function)
        if is_func_class:
            # Check for bare usage: word not followed by (
            # We must verify it's not being used properly elsewhere? 
            # Actually, just finding ONE bare usage is worth a warning usually.
            pattern = r"\b" + re.escape(name) + r"\b(?!\s*\()"
            if re.search(pattern, text):
                print(f"Error: '{name}' is a function class, did you mean '{name}(x)'?")
                found_issue = True
    
    if not found_issue:
        print("No bare function issue detected by heuristic.")

if __name__ == "__main__":
    test_parse()
