
import io
import contextlib
import sys

# Mock REPL class
class MockREPL:
    def __init__(self):
        self.variables = {}
        self.output_buffer = []
    
    def process_input(self, text):
        print(f"DEBUG: process_input called with '{text}'")
        # Simulate REPL assignment error if it reaches here
        if "=" in text:
            print("Error: LHS parse error: Unknown identifier...")

# Mock logic from streamlit_app.py
def test_shim(cli_input):
    print(f"\n--- Testing input: '{cli_input}' ---")
    repl_instance = MockREPL()
    output_buffer = []

    # Capture stdout logic
    with contextlib.redirect_stdout(io.StringIO()) as f:
        # Manual shim
        if cli_input.strip().lower().startswith("altv "):
                print(f"SHIM SUCCESS: Routing '{cli_input}' to _handle_evolve")
        else:
            # Run command - output goes to output_buffer via callback
            repl_instance.process_input(cli_input)
    
    std_out = f.getvalue()
    print("STDOUT CAPTURED:", std_out)

# Test cases
test_shim("altv f(4.5)=0.5")   # Standard
test_shim("altv  f(4.5)=0.5")  # Extra space
test_shim("altv")              # No args (Should fail shim, go to process_input)
test_shim("altv=5")            # Assignment (Should fail shim, go to process_input)
