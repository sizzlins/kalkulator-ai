
import sympy as sp
from kalkulator_pkg.parser import parse_expr, SAFE_GLOBALS

def test_implicit_solver():
    x, y = sp.symbols('x y')
    # Eq: cos(16(atan((x-2)/(y))+atan((y)/(x+2))))=0
    lhs_str = "cos(16*(atan((x-2)/y)+atan(y/(x+2))))"
    
    print(f"Testing equation: {lhs_str} = 0")
    
    try:
        expr = parse_expr(lhs_str, global_dict=SAFE_GLOBALS, local_dict={'x': x, 'y': y})
    except Exception as e:
        print(f"Parsing failed: {e}")
        return

    eq = sp.Eq(expr, 0)
    
    print("\n--- Attempt 1: Direct Solve for y ---")
    try:
        sols = sp.solve(eq, y)
        print(f"Solutions: {sols}")
    except NotImplementedError as e:
        print(f"Caught NotImplementedError: {e}")
    except Exception as e:
        print(f"Caught {type(e).__name__}: {e}")

    print("\n--- Attempt 2: Simplify then Solve ---")
    try:
        simplified_eq = sp.simplify(expr)
        print(f"Simplified Expression: {simplified_eq}")
        sols = sp.solve(sp.Eq(simplified_eq, 0), y)
        print(f"Solutions: {sols}")
    except Exception as e:
        print(f"Simplify failed: {e}")

    print("\n--- Attempt 3: TrigSimp then Solve ---")
    try:
        trig_eq = sp.trigsimp(expr)
        print(f"TrigSimp Expression: {trig_eq}")
        sols = sp.solve(sp.Eq(trig_eq, 0), y)
        print(f"Solutions: {sols}")
    except Exception as e:
        print(f"TrigSimp failed: {e}")

    print("\n--- Attempt 4: Simplify with inverse=True ---")
    try:
        # SymPy often refuses to simplify trig inverses without this hint
        simplified_inv = sp.simplify(expr, inverse=True)
        print(f"Inverse Simplify Expression: {simplified_inv}")
        sols = sp.solve(sp.Eq(simplified_inv, 0), y)
        print(f"Solutions: {sols}")
    except Exception as e:
        print(f"Inverse Simplify failed: {e}")

    print("\n--- Attempt 5: Unwrapping Trig ---")
    # Manually unwrap cos(ARG) = 0 -> ARG = pi/2 (limiting to principal branch for now)
    arg = expr.args[0]
    print(f"Argument: {arg}")
    # Solve arg = pi/2 + k*pi? Let's try just pi/2 first
    target = sp.pi/2
    print(f"Solving {arg} = {target}")
    try:
        sols = sp.solve(sp.Eq(arg, target), y)
        print(f"Solutions (Basic): {sols}")
    except Exception as e:
        print(f"Unwrap solve failed: {e}")

    print("\n--- Attempt 6: Simplify Argument ---")
    arg = expr.args[0]
    # Try simplifying the argument itself
    # Divide by 16 first to help simplification
    simple_arg = sp.simplify(arg)
    print(f"Simplified Arg: {simple_arg}")
    
    # Try trigsimp on the arg
    trig_arg = sp.trigsimp(arg)
    print(f"TrigSimp Arg: {trig_arg}")

    # Try manually combining atans: atan(a) + atan(b)
    # This checks if SymPy can do it if forced
    print("Trying solve with simplified args...")
    try:
        sols = sp.solve(sp.Eq(simple_arg, sp.pi/2), y)
        print(f"Sol (Simple Arg): {sols}")
    except Exception as e:
        print(f"Sol (Simple Arg) failed: {e}")
        
    try:
        sols = sp.solve(sp.Eq(trig_arg, sp.pi/2), y)
        print(f"Sol (TrigSimp Arg): {sols}")
    except Exception as e:
        print(f"Sol (TrigSimp Arg) failed: {e}")

if __name__ == "__main__":
    test_implicit_solver()

