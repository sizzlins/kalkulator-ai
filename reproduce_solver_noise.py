
import sys
import io
from kalkulator_pkg.solver.dispatch import solve_single_equation
from kalkulator_pkg.parser import parse_preprocessed

# Capture stdout/stderr to see if noise is printed
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

print("--- REPRODUCING SOLVER NOISE ---")
try:
    # Reproducing 'x^x+x-6=0' which caused "No algorithms are implemented..."
    # We simulate what the REPL does: call solve_single_equation on "x^x+x-6"
    # Note: REPL usually parses first.
    
    expr_str = "x^x+x-6"
    # Preprocessing handled by parser, but let's assume valid parse for now or use parser
    # Using the public API equivalent or internal dispatch
    
    result = solve_single_equation(expr_str)
    
    print(f"Result OK: {result.get('ok')}")
    print(f"Solutions: {result.get('exact', [])}")
    
except Exception as e:
    print(f"Exception Caught: {e}")

# Restore stdout
stdout_val = sys.stdout.getvalue()
stderr_val = sys.stderr.getvalue()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("CAPTURED STDOUT:", stdout_val)
print("CAPTURED STDERR:", stderr_val)
