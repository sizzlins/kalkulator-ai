import subprocess
import sys
import time

def run_e2e_test():
    print("--- STARTING E2E TEST OF kalkulator.py ---")
    
    # Inputs to feed
    inputs = [
        "f(-3) = 0, f(pi) = 6.1415926535\n",
        "solve x^2 - 2*y^2 - 1 = 0\n",
        "quit\n"
    ]
    
    process = subprocess.Popen(
        [sys.executable, "kalkulator.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0 # Unbuffered
    )
    
    try:
        stdout_data, stderr_data = process.communicate(input="".join(inputs), timeout=10)
        
        print("--- STDOUT ---")
        print(stdout_data)
        print("--- STDERR ---")
        print(stderr_data)
        
        # Validation
        success = True
        
        # 1. Regression Check
        if "1.0*x" in stdout_data or "x + 3.14" in stdout_data:
            print("\n[PASS] Regression: Found linear function.")
        elif "f(x) = 0.0" in stdout_data:
            print("\n[FAIL] Regression: Found Constant 0.0 (Symbolic Pi bug persists).")
            success = False
        else:
            print("\n[WARN] Regression: Output unclear.")
            
        # 2. Solve Command Check
        if "UnboundLocalError" in stderr_data or "UnboundLocalError" in stdout_data:
             print("[FAIL] Solve: UnboundLocalError crash.")
             success = False
        elif "x =" in stdout_data and "y =" in stdout_data:
             print("[PASS] Solve: Parametric solution found.")
        else:
             print("[FAIL] Solve: Solution not found or crashed.")
             success = False
             
        if success:
            print("\nOVERALL: SUCCESS")
        else:
            print("\nOVERALL: FAILURE")
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("\n[FAIL] Process timed out.")

if __name__ == "__main__":
    run_e2e_test()
