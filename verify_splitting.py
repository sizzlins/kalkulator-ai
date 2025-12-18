
from kalkulator_pkg.parser import split_top_level_commas

def test_splitting():
    print("Testing split_top_level_commas...")
    
    cases = [
        ("x=[1, 2, 3], y=[4, 5, 6]", ["x=[1, 2, 3]", "y=[4, 5, 6]"]),
        ("x=[1, 2], evolve f(x)", ["x=[1, 2]", "evolve f(x)"]),
        ("f(x)=min(x, y), g(x)=max(x, y)", ["f(x)=min(x, y)", "g(x)=max(x, y)"]),
        ("x=[1, 2, 3]", ["x=[1, 2, 3]"]), # Single item shouldn't split inside
    ]
    
    for input_str, expected in cases:
        result = split_top_level_commas(input_str)
        print(f"Input: '{input_str}'")
        print(f"Result: {result}")
        if result != expected:
            print(f"FAIL: Expected {expected}")
        else:
            print("PASS")

if __name__ == "__main__":
    test_splitting()
