
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Mock parser.split_top_level_commas
def mock_split_commas(text):
    # Poor man's implementation for test, or better: import real one if possible
    # But since we are mocking integration, let's try to import the real one?
    # No, let's mock the behavior we expect from repl_commands import
    # Wait, we want to test repl_commands logic which calls parser.split_top_level_commas
    # So we should allow it to import the real one or mock it properly.
    pass

# Mock evaluate_safely
sys.modules['kalkulator_pkg.worker'] = MagicMock()
# Return the input string as result for simplicity (simulating identity evaluation)
sys.modules['kalkulator_pkg.worker'].evaluate_safely = MagicMock(side_effect=lambda x: {"ok": True, "result": x})

# Mock function_manager
sys.modules['kalkulator_pkg.function_manager'] = MagicMock()

# Import repl_commands (this will try to import split_top_level_commas from ..parser)
# We need to make sure ..parser is available or mocked
sys.modules['kalkulator_pkg.parser'] = MagicMock()
# Mock the specific function
def real_split_logic(s):
    # Minimal logic to handle (1,2), (3,4)
    if s == "(1,2), (3,4)":
        return ["(1,2)", "(3,4)"]
    if s == "1,2":  # This is what inner becomes after stripping parens
        return ["1", "2"]
    if s == "3,4":
        return ["3", "4"]
    if s == "(a,b)":
         return ["a", "b"]
    return [s]

sys.modules['kalkulator_pkg.parser'].split_top_level_commas = MagicMock(side_effect=real_split_logic)

from kalkulator_pkg.cli import repl_commands

def test_callset_parsing():
    print("Testing callset command parsing...")
    
    # 1. Test Tuple Parsing
    # Command: callset grid (1,2), (3,4)
    # Expected: _CALL_SETS['grid'] = [('1', '2'), ('3', '4')] (assuming eval returns strings)
    
    cmd = "callset grid (1,2), (3,4)"
    repl_commands._handle_callset_command(cmd, None)
    
    grid = repl_commands._CALL_SETS.get("grid")
    print(f"Grid set: {grid}")
    
    assert grid is not None
    assert len(grid) == 2
    assert grid[0] == ('1', '2') # Mock evaluate_safely returns args as strings
    assert grid[1] == ('3', '4')
    
    print("Success!")

if __name__ == "__main__":
    test_callset_parsing()
