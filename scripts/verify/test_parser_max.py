import sympy as sp
import kalkulator_pkg.parser as kparser

print("Testing safe_sympy_parse for 'max'...")
try:
    x, y = sp.symbols('x y')
    local_dict = {'x': x, 'y': y}
    
    expr_str = "max(x, y)"
    print(f"Parsing: {expr_str}")
    
    # This should succeed now
    res = kparser.safe_sympy_parse(expr_str, local_dict=local_dict)
    
    print(f"Result: {res} (Type: {type(res)})")
    
    if isinstance(res, sp.Max):
        print("SUCCESS: Parsed as sp.Max")
    elif isinstance(res, sp.Function):
        # Could be Function("max") if fallback logic changed, but we expect sp.Max
        print(f"Result is Function: {res.func}")
    else:
        print(f"Result type unexpected: {type(res)}")
        
    # Check Min too
    expr_str_min = "min(x, y)"
    res_min = kparser.safe_sympy_parse(expr_str_min, local_dict=local_dict)
    print(f"Min result: {res_min}")
    
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
