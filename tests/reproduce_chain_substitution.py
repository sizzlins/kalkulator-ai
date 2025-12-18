import sys
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import re
import kalkulator_pkg.cli

# Debug print to verify module location

from kalkulator_pkg.cli import repl_loop

class TestChainSubstitution(unittest.TestCase):
    def test_chained_variable(self):
        commands = [
            "f(x)=2*x",
            "y=f(5), y-2^2",
            "quit"
        ]
        
        with patch('builtins.input', side_effect=commands), \
             patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            
            try:
                repl_loop()
            except SystemExit:
                pass
            
            output = mock_stdout.getvalue()
            print("Captured Output:\n", output)
            
            self.assertIn("Function 'f(x)' defined", output)
            self.assertIn("y = 10", output)
            # Check for correct chained calculation: 10 - 2^2 = 6
            self.assertIn("= 6", output)
            
            # Verify no phantom execution errors
            self.assertNotIn("Invalid syntax", output)
            self.assertNotIn("y-2^2 = y - 4", output)

if __name__ == "__main__":
    unittest.main()
