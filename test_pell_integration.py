from sympy import symbols, Eq
from kalkulator_pkg.solver.algebraic import solve_pell_equation_from_eq

def test_pell_integration():
    x, y = symbols("x y", integer=True)
    # x^2 - 2y^2 = 1
    eq = Eq(x**2 - 2*y**2, 1)
    
    print(f"Solving {eq}...")
    result = solve_pell_equation_from_eq(eq)
    print("Result:")
    print(result)
    
    if "x =" in result and "y =" in result and "t" in result:
        print("\nSUCCESS: Solution contains x, y, and parameter t.")
    else:
        print("\nWARNING: Unexpected format.")

if __name__ == "__main__":
    test_pell_integration()
