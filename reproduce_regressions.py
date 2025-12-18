
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.cli.app import repl_loop
from kalkulator_pkg.cli.context import ReplContext

def mock_input_generator(inputs):
    for i in inputs:
        yield i
    yield "quit"

def test_regressions():
    print("--- Testing Regressions ---")
    
    # Test 1: System Solver Output
    print("\nTest 1: System Solver")
    ctx = ReplContext()
    
    # We need to capture stdout
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        # Simulate 'x + y = 10, x - y = 2'
        # We can't easily call repl_loop with a generator for just one input and inspect internal state,
        # but we can check the output.
        # We need to bypass the infinite loop or throw an exception to exit?
        # repl_loop catches exceptions.
        # We'll rely on the generator yielding 'quit'.
        
        inputs = ["x + y = 10, x - y = 2"]
        # We need to mock input()
        
        # ACTUALLY, repl_loop takes no args and uses input().
        # We need to mock builtins.input.
        
        input_iter = mock_input_generator(inputs)
        
        def mock_input(prompt=None):
            try:
                return next(input_iter)
            except StopIteration:
                raise EOFError

        import builtins
        original_input = builtins.input
        builtins.input = mock_input
        
        try:
            repl_loop()
        except SystemExit:
            pass
        except EOFError:
            pass
        finally:
            builtins.input = original_input
            
        output = mystdout.getvalue()
        
        if "x = 6" in output and "y = 4" in output:
            print("SUCCESS: System solver output found.")
        else:
            print("FAILURE: System solver output MISSING.")
            print(f"Captured: {output}")

    except Exception as e:
        print(f"ERROR in Test 1: {e}")
    finally:
        sys.stdout = old_stdout
        print(mystdout.getvalue()) # Print mainly for debug

    # Test 2: --eval (Requires calling main or simulating args)
    # The --eval crash happens because chained_context is accessed when it doesn't exist.
    # We can inspect the code to see why.
    # But reproducing it via this script might be hard if we don't mock the argparse part.
    # Easier to verify fix by code inspection or running the actual command.

if __name__ == "__main__":
    test_regressions()
