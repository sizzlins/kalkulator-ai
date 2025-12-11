import sys
from io import StringIO
from unittest.mock import patch, MagicMock

# Import the specific function or module we want to test
# We want to test the REPL loop logic in app.py
# interacting with repl_loop is hard because it's a while True loop with sys.exit usually.
# However, we can mock input() to return our test string and then raise an exception or explicit exit input
# to break the loop, and capture stdout.

def test_formatting():
    print("--- Verifying Output Formatting ---")
    
    from kalkulator_pkg.cli.app import repl_loop
    from kalkulator_pkg.cli.context import ReplContext as Context

    # Inputs to simulate:
    # 1. "2+2, 4+4" -> Expected: "2+2 = 4, 4+4 = 8"
    # 2. "exit" -> To break the loop
    inputs = iter(["2+2, 4+4", "exit"])
    
    def mock_input(prompt=""):
        try:
            val = next(inputs)
            # print(f"DEBUG: Mock Input returning: {val}")
            return val
        except StopIteration:
            raise EOFError

    # Capture stdout
    captured_output = StringIO()
    
    # Mock context not needed as argument
    
    # Mock readline to avoid issues
    with patch('builtins.input', side_effect=mock_input), \
         patch('sys.stdout', captured_output): 
         
        # We need to catch SystemExit because 'exit' command raises it
        try:
            repl_loop()
        except SystemExit:
            pass
        except Exception as e:
            # repl_loop might catch exceptions, but let's be safe
            pass

    output = captured_output.getvalue()
    
    # Verify
    expected_fragment = "2+2 = 4, 4+4 = 8"
    if expected_fragment in output:
        print(f"SUCCESS: Found expected output: '{expected_fragment}'")
        return True
    else:
        print(f"FAILURE: Did not find expected output.")
        print(f"Captured Output:\n{output}")
        return False

if __name__ == "__main__":
    if test_formatting():
        sys.exit(0)
    else:
        sys.exit(1)
