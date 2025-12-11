import subprocess
import sys
import time

def drive_repl():
    print("Starting kalkulator.py subprocess...")
    # Force UTF-8 environment
    env = {"PYTHONIOENCODING": "utf-8"}
    
    # Use -u for unbuffered output to avoid deadlocks
    # Allow errors='replace' to handle encoding mismatches on Windows Console vs UTF-8
    process = subprocess.Popen(
        [sys.executable, "-u", "kalkulator.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace", # Critical fix for Windows console quirks
        cwd=".",
        bufsize=0 
    )

    inputs = [
        "calc 2^100\n",
        "x = 2 mod 3\n",
        "solve x^2 - 1 = 0\n",
        "h(x) = √(x^2 + 1)\n",
        "h(3)\n",
        "quit\n"
    ]

    print("Sending inputs...")
    try:
        # Write all inputs
        for cmd in inputs:
            process.stdin.write(cmd)
            process.stdin.flush()
            time.sleep(0.5) # Give it time to process
        
        # Get output
        stdout, stderr = process.communicate(timeout=10)
        
        print("\n--- REPL STDOUT ---")
        print(stdout)
        print("\n--- REPL STDERR ---")
        print(stderr)
        
        # Verify specific success markers
        failures = []
        if "1267650600228229964446656626688" not in stdout and "1.267" not in stdout:
             failures.append("calc 2^100 failed")
        if "x = 2" not in stdout:
             failures.append("mod operator failed")
        if "Exact: -1, 1" not in stdout and "x = -1, 1" not in stdout:
             failures.append("solve failed")
        # Unicode check:
        # The prompt echoes "Function 'h(x)' defined as: √(x^2 + 1)" if successful
        # Or evaluating h(3) gives approx 3.16
        if "3.1622" not in stdout:
             failures.append("Unicode sqrt definition failed (h(3) != 3.16)")
             
        if failures:
            print("\n[FAIL] REPL Check Failed:")
            for f in failures:
                print(f" - {f}")
            sys.exit(1)
        else:
            print("\n[PASS] All REPL interactive checks passed.")
            sys.exit(0)

    except Exception as e:
        print(f"Driver exception: {e}")
        process.kill()
        sys.exit(1)

if __name__ == "__main__":
    drive_repl()
