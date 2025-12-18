import sys
import os
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.worker import evaluate_safely

# Case from user
expr = "2.71828182845905^(sqrt(-1)*3.14159265358979)+1"

if __name__ == "__main__":
    from kalkulator_pkg.cache_manager import clear_cache_hits
    clear_cache_hits() # Ensure no previous hits interfere
    
    # Init worker manager
    from kalkulator_pkg.worker import _WORKER_MANAGER
    
    print(f"Evaluating: {expr}")
    res = evaluate_safely(expr)

    print(f"Result: {res}")

    if res.get("result") == "0":
        print("SUCCESS: Result snapped to 0")
    else:
        print(f"FAILURE: Result is {res.get('result')}")
        sys.exit(1)
        
    _WORKER_MANAGER.stop()
