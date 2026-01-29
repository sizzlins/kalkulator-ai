import sympy as sp
val = sp.sympify("-(-15)**(15/16)/15")
print(f"Type: {type(val)}")
evaluated = val.evalf()
print(f"Evaluated Type: {type(evaluated)}")
print(f"Evaluated: {evaluated}")
