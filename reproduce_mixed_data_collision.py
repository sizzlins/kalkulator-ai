import sympy as sp
from kalkulator_pkg.function_manager import find_function_from_data

def reproduction():
    """
    Reproduces the exact data state reported by the user:
    f(-3) = 0 (Integer/Float)
    f(pi) = 6.14... (Symbol("pi"))
    
    If 'pi' is not detected as numeric, the dataset splits:
    Numeric: [(-3, 0)] -> Length 1
    Symbolic: [(pi, 6.14...)] -> Length 1
    
    Result: Constant function (f(x)=0) due to lack of numeric points.
    """
    
    # Simulate data exactly as it comes from CLI parser
    # -3 comes as int/float
    # pi comes as Symbol because parsing 'f(pi)' makes 'pi' a symbol if not in context
    
    x1 = -3
    y1 = 0.0
    
    x2 = sp.Symbol("pi") # This is the key: CLI passes Symbol!
    y2 = 6.1415926535
    
    data = [
        ((x1,), y1),
        ((x2,), y2)
    ]
    
    print(f"Data Input: {data}")
    
    try:
        success, func, factored, msg = find_function_from_data(data, ["x"])
        print(f"Result: {func}")
        print(f"Msg: {msg}")
        
        if "1.0*x" in func or "x + 3.14" in func or "1.14" in func: # Linear fit
             # Note: f(-3)=0, f(pi)=6.14. 
             # Slope = (6.14 - 0)/(3.14 - (-3)) = 6.14/6.14 = 1.0
             # Intercept: 0 = 1*(-3) + b -> b = 3.
             # So f(x) = x + 3.
             if "x" in func and ("3.0" in func or "3." in func):
                 print("SUCCESS: Found linear function f(x) = x + 3")
             else:
                 print("PARTIAL SUCCESS: Found numeric function but maybe not exact linear.")
        elif "0.0" in func and "Constant" not in msg:
             print("FAILURE: Found Constant 0.0 (Fallback triggered incorrectly)")
        else:
             print("FAILURE: Something else found.")
             
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduction()
