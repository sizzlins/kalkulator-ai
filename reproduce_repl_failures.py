from kalkulator_pkg.parser import preprocess, REPL_COMMANDS
from kalkulator_pkg.cli.app import repl_loop
import sys
from unittest.mock import patch
from io import StringIO

def test_failures():
    print("Testing 'mod' operator preprocessing...")
    expr = "2 mod 3"
    pre = preprocess(expr)
    print(f"preprocess('{expr}') = '{pre}'")
    if "Mod(2, 3)" in pre or "2%3" in pre or "Mod(2,3)" in pre or "2 % 3" in pre: # or handled some other valid way
        print("PASS: mod handled")
    else:
        print("FAIL: mod not handled (likely implicit mul: 2*mod*3)")

    print("\nTesting 'calc' command...")
    # 'calc' should ideally be stripped or handled as a command
    # If not, it becomes c*a*l*c
    
    cmd = "calc 2^10"
    print(f"Command: '{cmd}'")
    
    # Check if 'calc' is in REPL_COMMANDS
    if "calc" in REPL_COMMANDS:
         print("INFO: 'calc' is in REPL_COMMANDS (might be blocked in preprocess)")
    else:
         print("INFO: 'calc' is NOT in REPL_COMMANDS")

    # Simulate basic check
    if cmd.startswith("calc "):
         stripped = cmd[5:]
         print(f"If handled, would evaluate: '{stripped}'")
    else:
         print("calc not handled specially")

if __name__ == "__main__":
    test_failures()
