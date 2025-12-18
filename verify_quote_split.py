
from kalkulator_pkg.parser import split_top_level_commas

def test_quote_split():
    print("Testing split_top_level_commas with quotes...")
    
    # Comma inside quotes should NOT split
    case = 'msg="Hello, world", x=1'
    expected = ['msg="Hello, world"', 'x=1']
    
    res = split_top_level_commas(case)
    print(f"Input: '{case}'")
    print(f"Result: {res}")
    
    if res != expected:
        print("FAIL: Quote protection missing.")
    else:
        print("PASS: Quote protection working.")

if __name__ == "__main__":
    test_quote_split()
