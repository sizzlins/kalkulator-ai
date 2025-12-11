import subprocess
import sys

def run_test():
    inputs = "f(-3) = 0, f(pi) = 6.1415926535\nsolve x^2 - 2*y^2 - 1 = 0\nquit\n"
    
    # Run Kalkulator
    process = subprocess.Popen(
        [sys.executable, "kalkulator.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8', 
        errors='ignore' 
    )
    
    print("Launching process...")
    stdout, stderr = process.communicate(input=inputs, timeout=30)
    
    print("\n--- APP STDOUT ---")
    print(stdout)
    print("--- APP STDERR ---")
    print(stderr)
    print("\n=== OUTPUT CHECK ===")
    
    # Check 1: Regression
    if "f(x) = 1.0*x + 3.0" in stdout or "3.0" in stdout:
        print("REGRESSION: PASS (Found linear)")
    elif "f(x) = 0.0" in stdout:
        print("REGRESSION: FAIL (Found Constant 0.0)")
    else:
        print(f"REGRESSION: UNKNOWN. Output snippet: {stdout[:200]}...")

    # Check 2: Solve
    if "x =" in stdout and "y =" in stdout:
        print("SOLVE: PASS (Found Parametric Solution)")
    elif "Error" in stderr or "Error" in stdout:
        print(f"SOLVE: FAIL (Error in output). Stderr: {stderr[:200]}")
    else:
        print("SOLVE: FAIL (No solution found)")

run_test()
