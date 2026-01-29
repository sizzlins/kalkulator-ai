
import sympy as sp
import numpy as np
import scipy

def test_evaluation():
    # Dataset provided by user
    x_val = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    y_true = np.array([-2, -1, -1, 0, 0, 1, 1, 2, 2, 3])
    
    print(f"X: {x_val}")
    print(f"Y (True): {y_true}")
    
    # Target expression: floor(trunc(x)/2)
    # Note: in sympy 'trunc' might not be standard in all contexts, usually it's floor/ceil or specific functions.
    # But we added 'trunc' to the allowed list and mapped it to np.trunc in the engine.
    
    x = sp.Symbol('x')
    
    # We need to manually construct the expression or parse it
    # Let's try to construct it using standard sympy functions where possible
    # Sympy doesn't have 'trunc' by default in the same way numpy does for expression trees sometimes
    # But let's assume we use the same custom dictionary as the engine
    
    # Expression 1: floor(x/2) - likely what the user meant if x is integer, but trunc is safer for negative
    expr1 = sp.floor(x / 2)
    
    # Expression 2: floor(trunc(x)/2) - using a custom Function for trunc if needed, or just assuming x is input to np.trunc
    # Since we can't easily sympify 'trunc' without defining it, let's define a dummy function
    expr2_str = "floor(trunc(x)/2)"
    
    # Setup the lambdify context similar to repl_commands.py
    # We need to make sure 'trunc' is available in the namespace
    
    # Mocking the custom modules from repl_commands.py
    def _v_primepi(x): 
        try: return float(sp.primepi(int(x))) 
        except: return 0.0
    def _v_prime(x): 
        try: return float(sp.prime(int(x))) 
        except: return 0.0

    custom_modules = [{
        "primepi": np.vectorize(_v_primepi), 
        "prime_pi": np.vectorize(_v_primepi), 
        "ith_prime": np.vectorize(_v_prime), 
        "prime": np.vectorize(_v_prime), 
        "trunc": np.trunc
    }, "numpy", "scipy"]
    
    # For expr2 (with trunc), we need to parse it or construct it. 
    # Since sympy doesn't have 'trunc', using sympify might fail unless we define it as a Function
    # OR we can just use sp.Function('trunc')(x)
    
    trunc = sp.Function('trunc')
    expr2 = sp.floor(trunc(x) / 2)
    
    print("\n--- Testing Expression 1: floor(x/2) ---")
    try:
        f1 = sp.lambdify(x, expr1, modules=custom_modules)
        y_pred1 = f1(x_val)
        print(f"Y (Pred): {y_pred1}")
        mse1 = np.mean((y_true - y_pred1)**2)
        print(f"MSE: {mse1}")
    except Exception as e:
        print(f"Error evaluating expr1: {e}")

    print("\n--- Testing Expression 2: floor(trunc(x)/2) ---")
    try:
        f2 = sp.lambdify(x, expr2, modules=custom_modules)
        y_pred2 = f2(x_val)
        print(f"Y (Pred): {y_pred2}")
        mse2 = np.mean((y_true - y_pred2)**2)
        print(f"MSE: {mse2}")
    except Exception as e:
        print(f"Error evaluating expr2: {e}")

if __name__ == "__main__":
    test_evaluation()
