from sympy.solvers.diophantine import diophantine
from sympy import symbols, Eq

def test_pell():
    x, y = symbols("x y", integer=True)
    # x^2 - 2y^2 = 1. Fundamental sol: (3, 2).
    # x^2 - 3y^2 = 1. Fundamental sol: (2, 1).
    # x^2 - 7y^2 = 1. Fundamental sol: (8, 3).
    
    eq1 = x**2 - 2*y**2 - 1
    print(f"Solving {eq1} = 0...")
    sols = diophantine(eq1)
    print(f"Result: {sols}")

    eq2 = x**2 - 7*y**2 - 1
    print(f"Solving {eq2} = 0...")
    sols = diophantine(eq2)
    print(f"Result: {sols}")

if __name__ == "__main__":
    test_pell()
