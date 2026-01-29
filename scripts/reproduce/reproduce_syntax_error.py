
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from kalkulator_pkg.solver.dispatch import solve_single_equation

def test_syntax_error():
    # Case 0: The one that allegedly works
    # eq0 = "x^3=x+x!"
    # print(f"\nTesting Equation (Reference): {eq0}")
    # try:
    #     res0 = solve_single_equation(eq0, find_var="x")
    #     print(f"Result ok: {res0.get('ok')}")
    #     if res0.get('ok'):
    #         print(f"Exact: {res0.get('exact')}")
    #         print(f"Approx: {res0.get('approx')}")
    #     else:
    #         print(f"Error: {res0.get('error')}")
    # except Exception as e:
    #     print(f"Reference crashed: {e}")

    # Case 1: Equation with == and factorials
    eq = "x^3==x!+x!"
    print(f"\nTesting Equation: {eq}")
    try:
        res = solve_single_equation(eq, find_var="x")
        print(f"Result ok: {res.get('ok')}")
        if not res.get('ok'):
            print(f"Error: {res.get('error')}")
        else:
            print(f"Solutions: {res.get('approx')}")
    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    test_syntax_error()
