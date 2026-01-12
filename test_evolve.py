"""Test evolving function to get full traceback."""
import sys
import os
sys.path.insert(0, os.getcwd())

import traceback

# Simulate the REPL
try:
    from kalkulator_pkg.cli import repl_core
    
    # Create a REPL instance
    repl = repl_core.REPL()
    
    # Run the problematic input
    input_str = "f(0.0)=1.0, f(0.5)=0.93847, f(1.0)=0.7652, f(1.5)=0.51183, evolve f(x)"
    print(f"Testing: {input_str}")
    
    try:
        result = repl.process_input(input_str)
        print(f"Result: {result}")
    except Exception as e:
        print("Error in process_line:")
        traceback.print_exc()
except Exception as e:
    print("Error creating REPL:")
    traceback.print_exc()
