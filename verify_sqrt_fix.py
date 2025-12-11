from kalkulator_pkg.function_manager import define_function, evaluate_function, clear_functions
import sympy as sp
import math

def verify_sqrt_fix():
    print("Testing Function Definition with Unicode Sqrt (√)...")
    
    # Clear any previous state
    clear_functions()
    
    # 1. Define function using Unicode sqrt
    func_name = "g"
    params = ["x"]
    body = "√(x^2 - 1)"  # This requires preprocessing to work!
    
    try:
        define_function(func_name, params, body)
        print(f"PASS: Defined {func_name}({params}) = {body}")
    except Exception as e:
        print(f"FAIL: define_function raised exception: {e}")
        return

    # 2. Evaluate with numeric input
    # g(1) = sqrt(0) = 0
    try:
        val = evaluate_function(func_name, [1])
        print(f"g(1) = {val} (Expected 0)")
        if val != 0:
            print("FAIL: g(1) incorrect")
        else:
            print("PASS: g(1) correct")
    except Exception as e:
        print(f"FAIL: Evaluation g(1) failed: {e}")

    # 3. Evaluate with Symbolic input (the user's failure case)
    # g(pi) should be sqrt(pi^2 - 1)
    # Previous failure: (pi^2 - 1) * √
    try:
        import sympy
        val = evaluate_function(func_name, [sympy.pi])
        print(f"g(pi) = {val}")
        
        # Check if result contains 'sqrt' function
        val_str = str(val)
        if "sqrt" in val_str and "pi**2" in val_str:
             print("PASS: Result contains sqrt and pi**2")
        elif "8.8696" in val_str and "sqrt" not in val_str.lower():
             # maybe evaluated to float?
             print(f"WARN: Result might be numeric: {val}")
        
        # Check specifically for the bug symptom: trailing * squareroot symbol or variable
        # The bug symptoms was '... * √' or '... * Symbol(√)'
        # In SymPy string, '√' variable would likely print as '√' or 'Symbol("√")'
        if "√" in val_str and "sqrt" not in val_str:
            print("FAIL: Result seems to contain raw √ variable/symbol!")
        else:
            print("PASS: Result does NOT contain raw √ variable.")
            
        # Numerical check
        numeric_val = val.evalf()
        expected = math.sqrt(math.pi**2 - 1)
        print(f"Numeric: {numeric_val} vs Expected: {expected}")
        if abs(numeric_val - expected) < 1e-9:
            print("PASS: Numerical match")
        else:
            print("FAIL: Numerical mismatch")

    except Exception as e:
        print(f"FAIL: Evaluation g(pi) failed: {e}")

if __name__ == "__main__":
    verify_sqrt_fix()
