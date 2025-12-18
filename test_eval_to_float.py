"""Test eval_to_float directly."""
import sys

sys.path.insert(0, ".")

# Import sympy first to set up the path properly
import sympy as sp

# Now test our function
local_ns = {
    'pi': sp.pi, 'e': sp.E, 'E': sp.E, 'I': sp.I,
    'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'log': sp.log, 'ln': sp.log, 'exp': sp.exp
}

test_values = ['0', 'pi', 'pi/2', 'e', '2*pi', 'sqrt(2)', '3.14']

for val in test_values:
    try:
        try:
            result = float(val)
        except ValueError:
            expr = sp.sympify(val, locals=local_ns)
            result_expr = expr.evalf()
            if hasattr(result_expr, 'free_symbols') and result_expr.free_symbols:
                print(f"'{val}' -> HAS FREE SYMBOLS: {result_expr.free_symbols}")
                continue
            result = float(result_expr)
        print(f"'{val}' -> {result}")
    except Exception as e:
        print(f"'{val}' -> ERROR: {e}")
