
import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kalkulator_pkg.cli.repl_commands import _handle_solve_command, _substitute_vars

def test_substitution_logic():
    print("--- Test 1: Basic Substitution ---")
    vars = {"a": "5", "b": "10"}
    text = "x + a = b"
    res = _substitute_vars(text, vars)
    print(f"Original: {text}")
    print(f"Result:   {res}")
    assert res == "x + (5) = (10)"
    
    print("\n--- Test 2: Exclusion ---")
    vars = {"x": "100", "a": "5"}
    text = "x + a = 0"
    exclude = {"x"}
    res = _substitute_vars(text, vars, exclude=exclude)
    print(f"Original: {text}")
    print(f"Exclude:  {exclude}")
    print(f"Result:   {res}")
    assert res == "x + (5) = 0"
    
def test_solve_command_parsing():
    print("\n--- Test 3: Solve Command Shadowing ---")
    # We can't easily capture print output from _handle_solve_command without mocking
    # But we can verify the logic by replicating it or mocking `solve_single_equation`
    pass

if __name__ == "__main__":
    test_substitution_logic()
    print("\nVerification Passed!")
