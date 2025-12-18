
import subprocess
import sys
import time
import os
import re

def run_challenges():
    print("=== Kalkulator Solver Challenges ===\n")
    
    cmd = [sys.executable, "kalkulator.py"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # 5 Distinct Challenges
    challenges = [
        ("Linear Trend", "f(1)=3, f(2)=5, f(3)=7"),
        ("Exponential Growth", "g(1)=2, g(2)=4, g(3)=8, g(4)=16"),
        ("Kinetic Energy (m=1)", "ke(1)=0.5, ke(2)=2.0, ke(3)=4.5"),
        ("ReLU (Logic)", "relu(-5)=0, relu(0)=0, relu(5)=5, relu(10)=10"),
        ("Logarithmic Scale", "lvl(1)=0, lvl(2)=1, lvl(4)=2, lvl(8)=3")
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
        env=env
    )
    
    import threading
    import queue
    
    output_queue = queue.Queue()
    
    def reader_thread():
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output_queue.put(line.strip())
    
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()
    
    def wait_for_discovery(timeout=20.0):
        start = time.time()
        result = None
        while time.time() - start < timeout:
            try:
                line = output_queue.get_nowait()
                # print(f"  [RAW] {line}")
                if "Discovered:" in line:
                    result = line
                if "No suitable model" in line:
                    result = "No model found."
            except queue.Empty:
                time.sleep(0.1)
        return result

    # Clear banner
    time.sleep(1)
    
    results = []
    
    try:
        for name, inp in challenges:
            print(f"Challenge: {name}")
            print(f"Input: {inp}")
            
            process.stdin.write(inp + "\n")
            process.stdin.flush()
            
            # Additional implicit "find" might be needed if auto-detect fails?
            # Current logic auto-detects if multiple points are given.
            
            res = wait_for_discovery(timeout=25.0)
            
            if res:
                print(f"Result: {res}\n")
                results.append((name, "PASS", res))
            else:
                print("Result: TIMEOUT or NO OUTPUT\n")
                results.append((name, "FAIL", "Timeout"))
                
        process.terminate()
        
        print("=== FINAL REPORT ===")
        for name, status, det in results:
            print(f"{name}: {status} -> {det}")
            
    except Exception as e:
        print(f"Error: {e}")
        try: process.kill()
        except: pass

if __name__ == "__main__":
    run_challenges()
