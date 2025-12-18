
import sys
import io
from contextlib import redirect_stdout

# Setup imports
try:
    from kalkulator_pkg.cli.repl_core import REPL
except ImportError:
    sys.path.append(".")
    from kalkulator_pkg.cli.repl_core import REPL

def test_repl_output():
    print("--- START REPL TEST ---")
    repl = REPL()
    
    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            # We add a small timeout or limit via mocking if needed, 
            # but for simple implicit search it should be fast.
            repl.process_input("x=[1, 2, 3], y=[2, 4, 6], evolve f(x)")
        except Exception as e:
            print(f"CRASH: {e}")
            
    output = f.getvalue()
    print("--- REPL OUTPUT CAPTURED ---")
    print(output)
    print("--- END REPL OUTPUT ---")

if __name__ == "__main__":
    test_repl_output()
