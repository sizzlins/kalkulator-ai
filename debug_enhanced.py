
import sympy as sp

print("Debugging Enhanced Solver Logic...")
x, y = sp.symbols('x y')
eq_lhs = sp.cos(16*(sp.atan((x-2)/y)+sp.atan(y/(x+2))))
equation = sp.Eq(eq_lhs, 0)
sym = y

def try_enhanced(equation, sym):
    try:
        print(f"Eq: {equation}")
        lhs_expr = equation.lhs
        rhs_val = equation.rhs
        
        new_eq = None
        if rhs_val == 0:
            if isinstance(lhs_expr, sp.cos):
                print("Detected cos=0")
                new_eq = sp.Eq(lhs_expr.args[0], sp.pi/2)
                
        if new_eq is not None:
            print(f"Unwrapped: {new_eq}")
            arg = new_eq.lhs
            arg = sp.factor(arg)
            print(f"Factored arg: {arg}")
            
            coeff = sp.S.One
            if isinstance(arg, sp.Mul):
                args_rest = []
                for term in arg.args:
                    if term.is_Number:
                        coeff *= term
                    else:
                        args_rest.append(term)
                arg = sp.Mul(*args_rest)
                new_eq = sp.Eq(arg, new_eq.rhs / coeff)
                print(f"Coeff {coeff}, Adjusted: {new_eq}")

            if isinstance(arg, sp.Add):
                atans = [t for t in arg.args if isinstance(t, sp.atan)]
                others = [t for t in arg.args if not isinstance(t, sp.atan)]
                print(f"Atans: {len(atans)}, Others: {len(others)}")
                
                if len(atans) == 2 and not others:
                    a, b = atans[0].args[0], atans[1].args[0]
                    print(f"a: {a}")
                    print(f"b: {b}")
                    combined_arg = (a + b) / (1 - a*b)
                    print(f"Combined Arg: {combined_arg}")
                    # combined_arg = sp.simplify(combined_arg) # Maybe simplify helps?
                    # print(f"Simplified Arg: {combined_arg}") # Simplify can be slow?
                    
                    tan_rhs = sp.tan(new_eq.rhs)
                    print(f"Tan RHS: {tan_rhs}")
                    final_eq = sp.Eq(combined_arg, tan_rhs)
                    print(f"Final Alg Eq: {final_eq}")
                    
                    sols = sp.solve(final_eq, sym)
                    print(f"Sols: {sols}")
                    return sols
    except Exception as e:
        print(f"Error: {e}")
    return None

sols = try_enhanced(equation, sym)
print(f"Result: {sols}")
