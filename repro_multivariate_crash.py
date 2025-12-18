
import subprocess
import sys
import os

def reproduce_crash():
    print("reproducing multivariate crash...")
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
    
    # Trigger the crash: multivariate data but "find m(x)" (1 var)
    inputs = [
        "m(1,1)=2, m(2,3)=5, m(10,5)=15, find m(x)"
    ]
    
    try:
        for inp in inputs:
            print(f">>> {inp}")
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            
        stdout, stderr = process.communicate(timeout=10)
        print("--- Output ---")
        print(stdout)
        print("--- Stderr ---")
        print(stderr)
        
        if "IndexError" in stdout or "IndexError" in stderr:
            print("SUCCESS: Configured crash reproduced.")
        else:
            print("FAILURE: Did not crash.")
            
    except Exception as e:
        print(f"Error: {e}")
        try: process.kill()
        except: pass

if __name__ == "__main__":
    reproduce_crash()
