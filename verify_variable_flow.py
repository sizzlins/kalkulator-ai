
from kalkulator_pkg.worker import evaluate_safely
import numpy as np
from kalkulator_pkg.cli.repl_commands import _handle_evolve

def test_flow():
    print("Testing evaluate_safely for list input...")
    
    # 1. Check evaluate_safely result for "[1, 2, 3]"
    input_str = "[2, 4, 6]" # Like 'y=[2, 4, 6]'
    res = evaluate_safely(input_str)
    
    print(f"evaluate_safely('{input_str}') result: {res}")
    
    if not res.get("ok"):
        print("FAIL: evaluate_safely failed.")
        return

    val_str = res.get("result")
    print(f"Result string: '{val_str}' (type: {type(val_str)})")
    
    # 2. Check if _handle_evolve parses this string correctly
    print("\nChecking _handle_evolve compatibility...")
    variables = {
        'y': val_str, # The string returned by evaluate_safely
        'x': "[1, 2, 3]" # Assume x worked similarly
    }
    
    # We want to see if 'y' lands in data_dict. 
    # Since we can't easily inspect data_dict without modifying code, 
    # we'll look at the debug prints I added (and removed, oh wait I logic-ed them out).
    # I'll rely on my previous knowledge of _handle_evolve logic:
    # it evals string variables if they contain "[" or "array".
    
    if "[" in val_str or "array" in val_str:
        print("PASS: Result string contains '[' or 'array', compatible with implicit loading.")
        
        # Try manual eval to be sure
        safe_dict = {"__builtins__": {}, "np": np, "array": np.array}
        try:
           val_eval = eval(val_str, safe_dict)
           arr = np.array(val_eval)
           print(f"Eval success: {arr} (dtype: {arr.dtype})")
           if arr.dtype.kind in 'iuf':
               print("PASS: Converted to numeric array.")
           else:
               print("FAIL: Array not numeric.")
        except Exception as e:
           print(f"FAIL: eval crashed: {e}")
    else:
        print("FAIL: Result string does NOT look like a list (no '[' or 'array'). Implicit loading will SKIP it.")

if __name__ == "__main__":
    test_flow()
