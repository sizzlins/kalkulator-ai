
import subprocess
import sys
import os

def repro_crash():
    print("Reproducing Constant Function Crash...")
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
    
    # Trigger the crash: f(1,2,3)=3, f(pi, e, i)=3, f(31,12,234)=3, find f(x,y,z)
    inp = "f(1,2,3)=3, f(3.14, 2.71, 1)=3, f(31,12,234)=3, find f(x,y,z)"
    
    print(f">>> {inp}")
    process.stdin.write(inp + "\n")
    process.stdin.flush()
            
    stdout, stderr = process.communicate(timeout=10)
    print("--- Output ---")
    print(stdout)
    print("--- Stderr ---")
    print(stderr)
    
    if "TypeError: 'NoneType' object is not iterable" in stdout or "TypeError: 'NoneType' object is not iterable" in stderr:
        print("SUCCESS: Crash reproduced.")
    else:
        print("FAILURE: Did not crash.")

if __name__ == "__main__":
    repro_crash()
