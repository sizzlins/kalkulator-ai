"""Reproduce the Function error."""
import sys
import os
sys.path.insert(0, os.getcwd())

# Simulate the REPL input that caused the error
input_text = "f(0.0)=1.0, f(0.5)=0.93847, f(1.0)=0.7652, f(1.5)=0.51183, f(2.0)=0.22389, f(2.5)=-0.04838, f(3.0)=-0.26005, f(3.5)=-0.38013, f(4.0)=-0.39715, f(4.5)=-0.32054, f(5.0)=-0.1776, f(5.5)=-0.00684, f(6.0)=0.15065, f(6.5)=0.26009, f(7.0)=0.30008, evolve f(x)"

try:
    from kalkulator_pkg.cli.repl_commands import process_command
    # Create a mock context
    class MockCtx:
        debug_mode = False
        timing_enabled = False
        show_cache_hits = False
    
    ctx = MockCtx()
    variables = {}
    
    result = process_command(input_text, ctx, variables)
    print(f"Result: {result}")
except Exception as e:
    import traceback
    print("FULL TRACEBACK:")
    traceback.print_exc()
