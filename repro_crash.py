
import sys
import os

# Set up path
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.cli.repl_core import REPL

def test_crash():
    repl = REPL()
    # Simulate the user input that caused the crash
    # Using a subset of points to keep it simple, but enough to trigger auto-detect
    # f(3,4)=4, f(5,6)=6 implies f(x,y)=y? No, f(3,4)=4. 
    # Just generic points.
    input_text = "f(3,4)=4, f(5,6)=6, f(7,8)=8"
    
    print(f"Processing input: {input_text}")
    try:
        repl.process_input(input_text)
        print("\n[SUCCESS] No crash detected.")
    except Exception as e:
        print(f"\n[FAIL] Crash detected: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crash()
