import sympy as sp
print(f"sp.Max type: {type(sp.Max)}")
print(f"sp.Function type: {type(sp.Function)}")
print(f"issubclass(sp.Max, sp.Function): {issubclass(sp.Max, sp.Function)}")
print(f"issubclass(sp.Max, sp.Expr): {issubclass(sp.Max, sp.Expr)}")
print(f"sp.Max bases: {sp.Max.__bases__}")
print(f"issubclass(sp.Piecewise, sp.Function): {issubclass(sp.Piecewise, sp.Function)}")
print(f"issubclass(sp.Heaviside, sp.Function): {issubclass(sp.Heaviside, sp.Function)}")
print(f"sp.Piecewise bases: {sp.Piecewise.__bases__}")
