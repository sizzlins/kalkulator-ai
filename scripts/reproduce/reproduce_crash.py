
import sympy as sp
from kalkulator_pkg.parser import parse_expr, SAFE_GLOBALS

print("Testing AccumBounds parsing...")
try:
    # This string simulates what happens when worker receives AccumBounds result
    test_str = "AccumBounds(-1, 1)"
    expr = parse_expr(test_str, global_dict=SAFE_GLOBALS)
    print(f"SUCCESS: Parsed '{test_str}' -> {expr} (Type: {type(expr).__name__})")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
