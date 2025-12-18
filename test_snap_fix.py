import sympy as sp
from sympy import I
import sys
import os
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.worker import _format_evaluation_result

# Case from user
# 2.71828182845905 is roughly e
e_val = sp.Float('2.71828182845905')
pi_val = sp.Float('3.14159265358979')
# expr: e^(i*pi) + 1
expr = e_val**(I * pi_val) + 1

print(f"Expr: {expr}")
result = _format_evaluation_result(expr)
print(f"Formatted result: {result}")

if result == "0":
    print("SUCCESS: Result snapped to 0")
else:
    print(f"FAILURE: Result is '{result}'")
    sys.exit(1)
