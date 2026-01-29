import subprocess
import sys
import os

os.chdir(r"c:\Users\LOQ\PycharmProjects\kalkulator-ai")

# Command sequence:
# 1. Run altvd with the Boyle's Law dataset
# 2. Evaluate f(10) to confirm it equals 4.0
cmd = "altvd f(10.0)=4.00, f(15.0)=2.67, f(20.0)=2.00, f(25.0)=1.60, f(30.0)=1.33, f(40.0)=1.00, f(50.0)=0.80, f(80.0)=0.50\nf(10)\nquit\n"

print("Running altvd command for Boyle's Law (40/x)...")
try:
    proc = subprocess.run(
        [sys.executable, "kalkulator.py"],
        input=cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=60
    )
    
    print("\n=== STDOUT ===")
    print(proc.stdout)
    
    if proc.stderr:
        print("\n=== STDERR ===")
        print(proc.stderr)
        
except Exception as e:
    print(f"Error: {e}")
