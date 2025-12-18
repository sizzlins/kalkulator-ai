
import sys
import unittest
import logging
import numpy as np

# Setup imports
try:
    from kalkulator_pkg.cli.repl_core import REPL
except ImportError:
    # Hack for running from root
    sys.path.append(".")
    from kalkulator_pkg.cli.repl_core import REPL

class TestChainEvolve(unittest.TestCase):
    def test_chain_evolve_implicit(self):
        print("Testing chain: x=[1,2,3], evolve f(x)")
        
        repl = REPL()
        test_input = "x=[1, 2, 3], y=[2, 4, 6], evolve f(x)"
        
        try:
            repl.process_input(test_input)
            print("PASS: process_input executed without exception.")
        except Exception as e:
            self.fail(f"process_input crashed: {e}")
            
        x_val = repl.variables.get('x')
        self.assertIsNotNone(x_val, "Variable x should be set")
        print(f"Variable x type: {type(x_val)}")
        
        if not isinstance(x_val, list):
             print(f"WARNING: x is {x_val}, updated logic should store list objects.")

if __name__ == "__main__":
    unittest.main()
