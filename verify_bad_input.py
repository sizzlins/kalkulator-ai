
from unittest.mock import MagicMock
from kalkulator_pkg.cli.repl_commands import _handle_evolve

def test_bad_input_logging():
    print("Testing Bad Input Logging...")
    
    # Mock variables with a "bad" y (object type, not numeric)
    # e.g. a list of strings that accidentally got created
    variables = {
        'x': [1, 2, 3],
        'y': ["a", "b", "c"], # Should fail 'arr.dtype.kind in iuf' check
        'z': [2.718, 5.436, 8.154] # Ghost variable
    }
    
    # Mock print to capture output
    # Since _handle_evolve prints to stdout
    # We'll just run it and look at the logs in the tool output
    
    try:
        # We need to monkeypatch handle_evolve's internal logic?
        # No, just calling it with bad variables should trigger the warning loop
        _handle_evolve("evolve f(x)", variables)
    except Exception as e:
        pass

if __name__ == "__main__":
    test_bad_input_logging()
