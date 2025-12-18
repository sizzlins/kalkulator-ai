
import subprocess
import sys
import time
import os

def test_repl_comments():
    print("Testing REPL Comments...")
    
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
        "# This is a comment",
        "2 + 2",
        " # Indented comment",
        "quit"
    ]
    
    try:
        for inp in inputs:
            print(f"Sending: {inp}")
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            time.sleep(0.5)
            
        stdout, stderr = process.communicate(timeout=5)
        
        print("\n--- Output ---")
        print(stdout)
        print("\n--- Error ---")
        print(stderr)
        
        # Check that 2+2 worked
        if "Result: 4" not in stdout:
             print("FAILURE: Calculation 2+2 failed.")
             sys.exit(1)
             
        # Check that comments didn't cause syntax errors
        if "SyntaxError" in stderr or "Invalid syntax" in stdout:
             # We need to be careful: did the comment trigger it?
             # The output buffer is mixed. Ideally we see NO error.
             if "Invalid syntax" in stdout: 
                  # Check if it was for the comment
                  if "# This is a comment" in stdout and "Invalid syntax" in stdout.split("# This is a comment")[1]:
                       print("FAILURE: Comment triggered invalid syntax.")
                       sys.exit(1)
        
        print("SUCCESS: REPL handled comments gracefully.")
            
    except Exception as e:
        print(f"FAILURE: {e}")
        process.kill()
        sys.exit(1)

if __name__ == "__main__":
    test_repl_comments()
