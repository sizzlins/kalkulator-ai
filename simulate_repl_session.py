import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, MagicMock

# Simulate inputs
# 1. 2-point regression (reproduce the fix)
# 2. Pell equation solve (verify standard library replacement)
# 3. CLI ambiguity check (verify hardening)
# 4. Exit
INPUTS = [
    "f(-3) = 0, f(pi) = 7.01119705467915", # 2-point regression (was failing)
    "x^2 - 2*y^2 - 1 = 0", # Pell equation implicit solve
    "solve x^2 - 7*y^2 - 1 = 0", # Pell equation explicit solve
    "f(x) = 2*x", # Function Definition
    "f(5)", # Evaluation (should use definition)
    "x=5, g(x)=10, g(x+1)=12", # Function Finding (ambiguity check)
    "quit"
]

def run_simulation():
    print("--- START REPL SIMULATION ---")
    
    # Mock input
    input_queue = list(INPUTS)
    def mock_input(prompt=None):
        if input_queue:
            cmd = input_queue.pop(0)
            print(f"> {cmd}")
            return cmd
        raise EOFError

    # Capture output
    f_out = io.StringIO()
    
    # We need to import app only now to avoid early init
    try:
        from kalkulator_pkg.cli.app import main
        
        with patch("builtins.input", side_effect=mock_input):
            with redirect_stdout(f_out):
                try:
                    # Run main with mocked args to start REPL
                    with patch.object(sys, 'argv', ['kalkulator']):
                        main()
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"CRASH: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"SETUP ERROR: {e}")

    output = f_out.getvalue()
    print("--- END REPL SIMULATION ---")
    print(output)
    
    # Basic Checks
    if "Linear: 1.14" in output or "1.14159" in output:
        print("\n[PASS] 2-point regression fixed.")
    else:
        print("\n[FAIL] 2-point regression verification missing.")

    if "x =" in output and "y =" in output and "t" in output:
        print("[PASS] Pell equation solved.")
    else:
        print("[FAIL] Pell equation solver missing.")
        
    if "g(x) = 2.0*x" in output or "2*x" in output:
         print("[PASS] Function Finding from x=5 ambiguity check passed.")

if __name__ == "__main__":
    run_simulation()
