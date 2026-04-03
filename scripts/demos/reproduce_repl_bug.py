
import sys
import os
import io

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kalkulator_pkg.cli.repl_core import REPL
from kalkulator_pkg.cli.context import ReplContext

# Capture output
class OutputCapture:
    def __init__(self):
        self.output = []
    
    def __call__(self, text):
        print(f"REPL: {text}")
        self.output.append(text)

def run_test():
    out = OutputCapture()
    ctx = ReplContext()
    repl = REPL(context=ctx, output_callback=out)
    
    print("\n--- Step 1: Initial sin(x)=cos(x) ---")
    repl.process_input("solve sin(x)=cos(x)")
    
    print("\n--- Step 2: Define x ---")
    repl.process_input("x = 10")
    
    print("\n--- Step 3: Clear x ---")
    repl.process_input("clear x")
    
    print("\n--- Step 4: sin(x)=cos(x) again ---")
    repl.process_input("solve sin(x)=cos(x)")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
