from kalkulator_pkg.parser import preprocess
import re

def test_sqrt_parsing():
    input_str = "√(x^2 - 1)"
    expected = "sqrt(x**2 - 1)"
    
    print(f"Input:    '{input_str}'")
    
    # 1. Test basic preprocess
    try:
        processed = preprocess(input_str)
        print(f"Processed: '{processed}'")
    except Exception as e:
        print(f"Preprocess Error: {e}")
        return

    # 2. Check if substitution happened
    if "sqrt(" in processed and "√" not in processed:
        print("PASS: √ converted to sqrt")
    else:
        print("FAIL: √ NOT converted (or incorrectly converted)")

    # 3. Check parentheses balance
    open_count = processed.count('(')
    close_count = processed.count(')')
    print(f"Parens:   {open_count} open, {close_count} close")
    
    if open_count != close_count:
        print("FAIL: Unbalanced parentheses")
    
    # 4. Check specific regex match
    from kalkulator_pkg.config import SQRT_UNICODE_REGEX
    match = SQRT_UNICODE_REGEX.search(input_str)
    print(f"Regex '{SQRT_UNICODE_REGEX.pattern}' match: {match}")

    # 5. Test with function definition context (simulating app.py)
    def_str = "g(x)=√(x^2 - 1)"
    try:
        # app.py splits first usually? Or preprocesses whole line?
        # Usually app.py parses 'g(x)' then 'preprocess' the rest.
        lhs, rhs = def_str.split('=')
        rhs_processed = preprocess(rhs)
        print(f"Def RHS:   '{rhs_processed}'")
    except Exception as e:
        print(f"Def Error: {e}")

if __name__ == "__main__":
    test_sqrt_parsing()
