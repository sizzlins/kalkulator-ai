import sympy as sp
import kalkulator_pkg.parser as kparser

print("Testing safe_sympy_parse for extended classes...")
try:
    x, y = sp.symbols('x y')
    local_dict = {'x': x, 'y': y}
    
    test_cases = [
        ("max(x, y)", sp.Max),
        ("min(x, y)", sp.Min),
        ("Piecewise((x, x > 0), (0, True))", sp.Piecewise),
        ("pow(x, 2)", sp.Pow),
        ("Eq(x, y)", sp.Eq),
        ("Gt(x, y)", sp.Gt),
        ("Matrix([[1, 2], [3, 4]])", sp.MatrixBase)
    ]
    
    for expr_str, expected_type in test_cases:
        try:
            print(f"Parsing: {expr_str}")
            res = kparser.safe_sympy_parse(expr_str, local_dict=local_dict)
            print(f"Result: {res} (Type: {type(res)})")
            
            if isinstance(res, expected_type) or issubclass(type(res), expected_type):
                 print(f"SUCCESS: Parsed as {expected_type.__name__}")
            else:
                 print(f"FAILURE: Expected {expected_type.__name__}, got {type(res)}")
                 
        except Exception as e:
            print(f"FAILURE on '{expr_str}': {e}")
            if "Calling symbolic variables" in str(e):
                print(" -> Still rejected by sanitizer!")

except Exception as e:
    print(f"GLOBAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
