
import subprocess
import sys
import os

def reproduce_evolve_error():
    print("Reproducing evolve arity mismatch...")
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
    
    # 1. Define 2D data IN the command to test parsing logic
    inputs = [
        "evolve m(x) from m(1,1)=2, m(2,3)=5",
        "quit"
    ]
    
    try:
        for inp in inputs:
            print(f">>> {inp}")
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            
        stdout, stderr = process.communicate(timeout=5)
        print("--- Output ---")
        print(stdout)
        
        if "LHS parse error" in stdout or "expects 1 argument" in stdout:
            print("SUCCESS: Reproduction confirmed.")
        else:
            print("FAILURE: Did not reproduce specific error.")
            
    except Exception as e:
        print(f"Error: {e}")
        try: process.kill()
        except: pass

if __name__ == "__main__":
    reproduce_evolve_error()
