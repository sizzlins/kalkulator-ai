
from sympy import symbols, sqrt, I
from kalkulator_pkg.parser import parse_preprocessed
from kalkulator_pkg.utils.formatting import format_solution

x, y = symbols('x y')

print("--- VERIFYING FORMATTING FIXES ---")

# Case 1: y = x*(x+1) -> wanted x(x+1)
expr1 = x*(x+1)
result1 = format_solution(str(expr1))
print(f"Expr 1 (Raw): {str(expr1)}")
print(f"Expr 1 (Fmt): {result1}")
assert "x(x + 1)" in result1 or "x*(x + 1)" not in result1, f"Failed implicit mul: {result1}"

# Case 2: -sqrt(4*y + 1)/2 - 1/2
# SymPy output: -sqrt(4*y + 1)/2 - 0.5
expr2 = -sqrt(4*y + 1)/2 - 0.5
result2 = format_solution(str(expr2))
print(f"Expr 2 (Raw): {str(expr2)}")
print(f"Expr 2 (Fmt): {result2}")
assert "√" in result2, "Failed sqrt conversion"
assert "**" not in result2, "Failed exponent conversion"

# Case 3: Imaginary numbers
# SymPy output often 1.0*I or I*1.0
expr3 = 1.0*I
result3 = format_solution(str(expr3))
print(f"Expr 3 (Raw): {str(expr3)}")
print(f"Expr 3 (Fmt): {result3}")
# Logic says 1.0*I -> 1*I -> i in format_solution?
# Let's check format_solution manual logic in reproduction

# Case 4: Power
expr4 = x**2
result4 = format_solution(str(expr4))
print(f"Expr 4 (Raw): {str(expr4)}")
print(f"Expr 4 (Fmt): {result4}")
assert "^" in result4, "Failed power conversion"
