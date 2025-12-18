
import subprocess
import sys
import os

def repro_commands():
    print("Reproducing Utility Commands Issues...")
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
    
    # Sequence:
    # 1. timing on -> 2+2 (Expect "Execution time: ...")
    # 2. showcache (Expect list of items?)
    # 3. showcachehits on (Expect toggle confirmation)
    inp = "timing on\n2+2\nshowcache\nquit\n"
    
    print(f">>> {inp}")
    process.stdin.write(inp)
    process.stdin.flush()
            
    stdout, stderr = process.communicate(timeout=10)
    print("--- Output ---")
    print(stdout)
    
    if "Execution time:" in stdout or "seconds" in stdout:
        print("TIMING: FOUND")
    else:
        print("TIMING: MISSING")
        
    if "2+2" in stdout and ("Cache contains" in stdout):
         # We want to see if it lists keys. If it just says "Cache contains 227 items", that's what user complained about.
         pass

if __name__ == "__main__":
    repro_commands()
