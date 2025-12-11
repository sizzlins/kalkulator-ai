import sympy as sp
from kalkulator_pkg import parser

expr = parser.parse_preprocessed("pi")
print(f"Type: {type(expr)}")
print(f"Repr: {repr(expr)}")
print(f"Assumptions: {expr.assumptions0 if hasattr(expr, 'assumptions0') else 'N/A'}")
print(f"Equality check: {expr == sp.Symbol('pi')}")

# Check subs
subbed = expr.subs({sp.Symbol('pi'): sp.pi})
print(f"Subbed repr: {repr(subbed)}")
print(f"Is floatable? {subbed.is_number}")
try:
    print(f"Float value: {float(sp.N(subbed))}")
except Exception as e:
    print(f"Float fail: {e}")
