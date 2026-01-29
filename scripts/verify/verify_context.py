
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from kalkulator_pkg.core import Context
    from kalkulator_pkg.function_manager import define_function, evaluate_function, list_functions
except ImportError as e:
    print(f"FAIL: Import error: {e}")
    sys.exit(1)

def test_context_isolation():
    print("Testing Context Isolation (No Global State)...")
    
    # Context A
    ctx_a = Context()
    define_function(ctx_a, "f", ["x"], "x + 1")
    print("  [Context A] Defined f(x) = x + 1")
    
    # Context B
    ctx_b = Context()
    define_function(ctx_b, "f", ["x"], "x * 10")
    print("  [Context B] Defined f(x) = x * 10")
    
    # Verify A
    res_a = evaluate_function(ctx_a, "f", [5])
    print(f"  [Context A] f(5) = {res_a}")
    if res_a != 6:
        print(f"FAIL: Context A returned {res_a}, expected 6")
        return False
        
    # Verify B
    res_b = evaluate_function(ctx_b, "f", [5])
    print(f"  [Context B] f(5) = {res_b}")
    if res_b != 50:
        print(f"FAIL: Context B returned {res_b}, expected 50")
        return False
        
    # Verify no leak
    funcs_a = list_functions(ctx_a)
    funcs_b = list_functions(ctx_b)
    
    if len(funcs_a) != 1 or len(funcs_b) != 1:
        print("FAIL: Function counts incorrect")
        return False
        
    print("PASS: Contexts are isolated.")
    return True

if __name__ == "__main__":
    if test_context_isolation():
        print("\nSUCCESS: Global state successfully removed via Context Passing.")
        sys.exit(0)
    else:
        print("\nFAILURE: Context isolation failed.")
        sys.exit(1)
