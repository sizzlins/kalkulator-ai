
import multiprocessing
import sympy as sp
from kalkulator_pkg.solver.dispatch import solve_single_equation

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Testing Implicit Trig Solver...")

    # Eq: cos(16(atan((x-2)/y)+atan(y/(x+2)))) = 0
    eq_str = "cos(16*(atan((x-2)/y)+atan(y/(x+2)))) = 0"

    # Note: For testing dispatch, we simulate single equation solving
    # The dispatch module's solve_single_equation is for single variable or general eq
    # However, usually the CLI calls specific solvers or evaluate-and-solve.
    # We will test solve_single_equation solving for 'y' assuming 'x' is a symbol

    # But solve_single_equation implementation handles "find_var"
    # Let's try to solve for 'y' treating 'x' as allowed symbolic constant (not finding it)
    result = solve_single_equation(eq_str, find_var="y")

    if result["ok"]:
        print(f"SUCCESS: Found solutions: {result.get('exact') or result.get('solutions')}")
        # Verify approximate numeric output
        if 'approx' in result and result['approx']:
            print(f"Approx: {result['approx']}")
    else:
        print(f"FAIL: {result.get('error')}")

    # Also test the generic case without specific var to see multi-isolate behavior
    # This requires slight hack as solve_single_equation expects single var or none (eval)
    # Actually, solve_single_equation logic branches:
    # if find_var: ...
    # else: if len(symbols)==1: ... else: multi_isolate
    result_multi = solve_single_equation(eq_str) # No find_var, multiple symbols -> multi_isolate
    if result_multi["ok"] and result_multi["type"] == "multi_isolate":
        print("SUCCESS: Multi-isolate returned:")
        for var, sols in result_multi["solutions"].items():
            print(f"  {var}: {sols}")
    else:
        print(f"FAIL Multi: {result_multi.get('error')}")
