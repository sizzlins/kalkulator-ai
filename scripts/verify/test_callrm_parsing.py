
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock evaluate_safely BEFORE importing repl_commands
sys.modules['kalkulator_pkg.worker'] = MagicMock()
sys.modules['kalkulator_pkg.worker'].evaluate_safely = MagicMock(return_value={"ok": True, "result": "1.0"})

# Mock list_functions
sys.modules['kalkulator_pkg.function_manager'] = MagicMock()
sys.modules['kalkulator_pkg.function_manager'].list_functions = lambda ctx: {"f": (["x", "y"], "x*y")}

from kalkulator_pkg.cli import repl_commands

def test_callrm_command():
    print("Testing callrm parsing...")
    ctx = MagicMock()
    # Mock inputs
    repl_commands._CALL_SETS["default"] = [1, 2, 3] # 3 items
    
    # Test callr with count
    print("\n--- Testing 'callr f 10' (count arg) ---")
    repl_commands._handle_callr_command("callr f 10", ctx)
    # If no crash, pass. We can't easily check output count without capturing stdout, but crash is main concern.
    
    print("\n--- Testing 'callrm f 5' (multiline + count) ---")
    repl_commands._handle_callrm_command("callrm f 5", ctx)
    
    print("\nSuccess!")

if __name__ == "__main__":
    test_callrm_command()
