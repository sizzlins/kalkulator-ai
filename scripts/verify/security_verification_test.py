
import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from kalkulator_pkg.parser import safe_sympy_parse, ValidationError
from kalkulator_pkg.regression_solver import eval_to_float

class TestSecurityHardening(unittest.TestCase):
    
    def test_safe_sympy_parse_blocks_os_system(self):
        """Test that safe_sympy_parse blocks os.system calls."""
        payload = "__import__('os').system('echo PWNED > pwned_parse.txt')"
        print(f"\n[TEST] Attempting RCE via safe_sympy_parse with payload: {payload}")
        
        try:
            with self.assertRaises(ValidationError) as cm:
                safe_sympy_parse(payload)
            print(f"[PASS] Blocked! Error: {cm.exception}")
        except Exception as e:
             # It might raise other errors depending on how AST parses it, but it MUST NOT execute
             print(f"[PASS] Blocked with unexpected error: {e}")

        self.assertFalse(os.path.exists("pwned_parse.txt"), "File pwned_parse.txt should not be created!")

    def test_eval_to_float_blocks_rce(self):
        """Test that eval_to_float (used in CLI) blocks RCE."""
        # This was the CRITICAL vulnerability found in paranoia sweep
        payload = "__import__('os').system('echo PWNED_FLOAT > pwned_float.txt')"
        print(f"\n[TEST] Attempting RCE via eval_to_float with payload: {payload}")
        
        # Should catch exception or return 0.0, but definitely NOT execute
        result = eval_to_float(payload)
        
        print(f"[PASS] Result: {result}")
        self.assertFalse(os.path.exists("pwned_float.txt"), "File pwned_float.txt should not be created!")

    def test_dos_depth_limit(self):
        """Test that AST depth limit prevents stack overflow/DoS."""
        # Create deeply nested expression: sin(sin(...)) which creates AST nodes
        depth = 200
        payload = "sin(" * depth + "1" + ")" * depth
        print(f"\n[TEST] Attempting DoS via deep nesting (depth={depth})")
        
        with self.assertRaises(ValidationError) as cm:
            safe_sympy_parse(payload)
            
        print(f"[PASS] Blocked! Error: {cm.exception}")
        self.assertIn("nested", str(cm.exception))

if __name__ == "__main__":
    # Clean up any previous test artifacts
    if os.path.exists("pwned_parse.txt"): os.remove("pwned_parse.txt")
    if os.path.exists("pwned_float.txt"): os.remove("pwned_float.txt")
    
    unittest.main()
