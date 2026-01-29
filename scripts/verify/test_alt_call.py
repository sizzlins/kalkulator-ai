
import sys
import os
import re
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock evaluate_safely BEFORE importing repl_commands
sys.modules['kalkulator_pkg.worker'] = MagicMock()
sys.modules['kalkulator_pkg.worker'].evaluate_safely = MagicMock(return_value={"ok": True, "result": "1.0"})

# Mock function_manager
sys.modules['kalkulator_pkg.function_manager'] = MagicMock()
sys.modules['kalkulator_pkg.function_manager'].list_functions = lambda ctx: {"f": (["x", "y"], "x*y")}

# Mock other deps
sys.modules['kalkulator_pkg.cli.app'] = MagicMock()
sys.modules['kalkulator_pkg.symbolic_regression.evolve'] = MagicMock()

from kalkulator_pkg.cli import repl_commands

# Initialize context and call sets
ctx = MagicMock()
repl_commands._CALL_SETS["default"] = [1, 2, 3]

def test_alt_call():
    print("Testing 'altv1 callr f 10'...")
    
    # We need to capture the output effectively or just ensure it calls handle_command recursively
    # We can mock handle_command to print the command it receives
    original_handle = repl_commands.handle_command
    
    def mock_recursive_handle(cmd, ctx, vars):
        print(f"Recursive call: {cmd[:50]}...")
        return True
        
    # We only want to mock the RECURSIVE call, but we are testing the OUTER call.
    # The outer call logic is in handle_command.
    # So we can't mock handle_command easily if we call it.
    # But wait, lines 222: `return handle_command(new_cmd, ctx, variables)`
    # This calls `handle_command` again.
    
    # We can patch `repl_commands.handle_command` inside the module?
    # No, it calls itself recursively?
    # It calls `handle_command` (global function reference).
    
    # Let's just run it and see the output.
    # We expect `[AltCall] ...`
    # And then recursive call `altv1 ... f(..)=..`
    # The recursive call will hit `_handle_evolve` (the logic later in the function).
    # We can mock `_handle_evolve` to verify it gets called with data.
    
    repl_commands._handle_evolve = MagicMock()
    
    cmd = "altv1 callr f 10"
    repl_commands.handle_command(cmd, ctx, {})
    
    # Verify _handle_evolve was called
    if repl_commands._handle_evolve.called:
        args = repl_commands._handle_evolve.call_args[0]
        text_arg = args[0]
        print(f"Success! _handle_evolve called with text length: {len(text_arg)}")
        if "f(2, 1) = 1.0" in text_arg:
             print("Data points found in recursive call.")
        else:
             print("FAIL: No data points in recursive call.")
             print(f"Text content: {text_arg[:100]}...")
    else:
        print("FAIL: _handle_evolve NOT called. Recursion likely failed.")

if __name__ == "__main__":
    test_alt_call()
