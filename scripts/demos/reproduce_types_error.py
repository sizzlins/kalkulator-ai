
import sympy as sp

try:
    print("Testing sp.sin(sp.Symbol('x'))")
    print(sp.sin(sp.Symbol('x')))
except Exception as e:
    print(f"Error 1: {e}")

try:
    print("Testing sp.sin(sp.Expr)")
    print(sp.sin(sp.Expr))
except TypeError as e:
    print(f"Error 2: {e}")
except Exception as e:
    print(f"Error 2 (Other): {type(e).__name__}: {e}")

try:
    print("Testing sp.sin(sp.core.expr.Expr)")
    print(sp.sin(sp.core.expr.Expr))
except TypeError as e:
    print(f"Error 3: {e}")
    
# Check what else could interpret 'x'
class MockX:
    def could_extract_minus_sign(self):
        return True
        
try:
    print("Testing sp.sin(MockX())")
    print(sp.sin(MockX()))
except Exception as e:
    print(f"Error 4: {e}")

# Check Class with method but called as class
try:
    print("Testing sp.sin(MockX) - Class")
    print(sp.sin(MockX))
except TypeError as e:
    print(f"Error 5: {e}")
