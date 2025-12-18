
import subprocess
import sys
import time
import os

def test_func_def():
    print("Testing Function Definition & Call Persistence...")
    cmd = [sys.executable, "kalkulator.py"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env=env
    )
    
    inputs = [
        "f(x) = x**2",
        "f(10)",
        "showfunctions",
        "quit"
    ]
    
    try:
        for inp in inputs:
            print(f">>> {inp}")
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            time.sleep(0.5)
            
        stdout, stderr = process.communicate(timeout=5)
        print("--- Output ---")
        print(stdout)
        
        if "Result: 100" in stdout:
            print("SUCCESS: Function called correctly.")
        else:
            print("FAILURE: Function call failed.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        process.kill()

if __name__ == "__main__":
    test_func_def()
