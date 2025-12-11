
"""
Reproduction script for parser error on undefined functions.
Steps:
1. Try to parse "f(pi)=0" where 'f' is not defined.
2. Expect ValidationError: "Undefined function 'f'".
"""
import sys
import logging
from kalkulator_pkg.parser import parse_preprocessed, ValidationError

# Mock config to ensure 'f' is NOT in implicit globals if applicable
# (But defaults should show it's not defined)

def test_parse_undefined_function():
    expr = "f(pi)=0"
    print(f"Testing parse: '{expr}'")
    try:
        # We assume preprocess happens before parse, but parse_preprocessed takes string.
        # Wait, 'parse_preprocessed' expects preprocessed string?
        # CLI calls process_input -> ... -> eventually parses.
        # Let's try raw parse first.
        # Note: "f(pi)=0" is an equation. parse_preprocessed might return an Eq object or handle inputs.
        # In parser.py, we usually parse expressions. Equations are handled in 'solver'.
        # Let's see what the user error says: "LHS parse error: Undefined function 'f'".
        # This implies it parses LHS "f(pi)" separately.
        
        lhs = "f(pi)"
        ast_tree = parse_preprocessed(lhs)
        print(f"Successfully parsed: {ast_tree}")
        
    except ValidationError as e:
        print(f"Caught expected ValidationError: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_parse_undefined_function()
