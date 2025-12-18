
import subprocess
import sys
import time
import os
import re

def run_full_suite():
    print("=== Kalkulator Engine: Comprehensive Capability Test ===\n")
    
    cmd = [sys.executable, "kalkulator.py"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Increase timeout for discovery tasks
    # Set seed if possible? No, we rely on robustness.
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env=env
    )
    
    # (Category, Input, Expected Pattern)
    tests = [
        # --- Basics ---
        ("Arithmetic", "2 + 2", r"Result: 4"),
        ("Floats", "10 / 4", r"Result: 2\.5"),
        ("Complex", "sqrt(-1)", r"Result: I"),
        
        # --- UX / Parsing ---
        ("Comments", "# This should be ignored", None), # Should not error
        ("Implicit Mult", "2x + 3x assuming x=10", None), # Hard to test value without finding 'x', but parsing shouldn't crash
        # Actually simplest implicit mult test:
        ("Implicit Mult Simple", "2(3)", r"Result: 6"),
        ("Log2 Parsing", "log2(32)", r"Result: 5"),
        
        # --- Symbolic Algebra & Calculus ---
        ("Expand", "expand((x+1)**2)", r"x\*\*2 \+ 2\*x \+ 1|x\^2 \+ 2x \+ 1"),
        ("Diff", "diff(x**2, x)", r"Result: 2\*x"),
        ("Integrate", "integrate(2*x, x)", r"x\*\*2"),
        ("Limit", "limit(sin(x)/x, x, 0)", r"Result: 1"),
        
        # --- Logic & Special Functions ---
        ("Min/Max", "min(10, 20)", r"Result: 10"),
        ("Inequality", "10 > 5", r"True|Result: True"),
        
        # --- Variables & Functions ---
        ("Var Assign", "my_var = 123", r"\(my_var\) saved"),
        ("Var Use", "my_var * 2", r"Result: 246"),
        ("Func Def", "my_f(x) = x^2", r"Function my_f defined"),
        ("Func Call", "my_f(10)", r"Result: 100"),
        
        # --- DISCOVERY (The Heavy Hitters) ---
        # 1. Linear
        ("Disc Linear", "d1(0)=0; d1(1)=2; d1(2)=4; d1(3)=6", r"Discovered: .*d1\(x\)\s*=\s*(2\.0\*x|2\*x)"),
        
        # 2. Physics (Free Fall approx 4.9*t^2)
        ("Disc Physics", "dist(0)=0.0; dist(1)=4.9; dist(2)=19.6; dist(3)=44.1", r"4\.9\*x\*\*2|4\.9\*x\^2"),
        
        # 3. Logarithmic (Testing log2 support in discovery)
        ("Disc Log", "dl(1)=0; dl(2)=1; dl(4)=2; dl(8)=3", r"log\(x\)/log\(2\)|log\(x, 2\)"),
        
        # 4. Multivariate
        ("Disc Multi", "dm(1,1)=2; dm(2,3)=5; dm(10,5)=15", r"x \+ y"),
    ]
    
    results = {}
    
    try:
        # We need to read stdout continuously to match patterns
        # But for simplicity in this script, we'll send inputs and buffer read logic.
        # Actually mixed buffer reading is hard. 
        # Better strategy: Send one, wait for specific marker or timeout? 
        # The REPL prints ">>> " prompts.
        
        # We will use a dedicated read loop.
        
        import threading
        import queue
        
        output_queue = queue.Queue()
        
        def reader_thread():
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # print(f"[DEBUG RAW] {line.strip()}") 
                    output_queue.put(line.strip())
        
        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()
        
        def get_all_output(timeout=1.0):
            buffer = []
            start = time.time()
            while time.time() - start < timeout:
                try:
                    line = output_queue.get_nowait()
                    buffer.append(line)
                    start = time.time() # Reset timeout logic on activity? No, absolute/short timeout
                except queue.Empty:
                    time.sleep(0.05)
            return "\n".join(buffer)

        # Clear initial banner
        get_all_output(timeout=2.0)
        
        failures = []
        
        for name, inp, expected in tests:
            print(f"TESTING: {name} [{inp}]")
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            
            # Variable timeout depending on complexity
            if "Disc" in name:
                wait_time = 20.0 # Give it time to evolve
            else:
                wait_time = 2.0  # Increased for stability
                
            time.sleep(0.2) # Small flush wait
            output = get_all_output(timeout=wait_time)
            # print(f"--- Output ---\n{output}\n----------------")
            
            if expected:
                if re.search(expected, output, re.IGNORECASE):
                    print(f" PASS: {name}")
                else:
                    print(f" FAIL: {name} (Expected match for '{expected}')")
                    print(f"       Got: {output.replace(chr(10), ' ')}")
                    failures.append(name)
            else:
                # Expect NO error
                if "Error" in output or "Exception" in output or "Traceback" in output:
                     print(f" FAIL: {name} (Crashed)")
                     print(f"       Got: {output}")
                     failures.append(name)
                else:
                    print(f" PASS: {name}")

        process.stdin.write("quit\n")
        process.stdin.flush()
        process.terminate()
        
        print("\n=== SUMMARY ===")
        if failures:
            print(f"FAILED TESTS ({len(failures)}):")
            for f in failures:
                print(f"- {f}")
            sys.exit(1)
        else:
            print("ALL SYSTEMS OPERATIONAL.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        try:
            process.kill()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    run_full_suite()
