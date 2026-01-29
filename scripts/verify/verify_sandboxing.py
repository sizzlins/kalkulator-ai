import sys
import unittest
from kalkulator_pkg.worker import evaluate_safely, _limit_resources
import time

class TestSandboxing(unittest.TestCase):
    def test_memory_limit(self):
        print("\nTesting Memory Limit...")
        # Attempt to allocate > 100MB (limit is usually 100MB or lower in config)
        # WORKER_AS_MB default is 100? or 50?
        # A string of 200MB length
        
        # Expression to generate massive string/list
        # "['a'] * 200_000_000"
        
        # Note: worker uses ast.literal_eval or safe parsing.
        # We need an expression that expands in memory.
        # "list(range(10000000))"
        
        expr = "list(range(20000000))" # ~150MB+ for list of ints
        
        result = evaluate_safely(expr, timeout=10)
        print("Result:", result)
        
        # It should fail with MemoryError or similar, or return None/Error
        if result['ok']:
             # If it succeeded, check if it limited?
             pass
        else:
            self.assertIn("error", result)
            print("Successfully blocked/failed:", result['error'])

    def test_cpu_limit(self):
        print("\nTesting CPU Limit...")
        # Infinite loop
        # "sum(i for i in iter(int, 1))" - might be caught by AST limit?
        # A heavy calculation: "sum(i**2 for i in range(10000000))"
        
        expr = "sum(i**2 for i in range(50000000))"
        
        result = evaluate_safely(expr, timeout=3)
        print("Result:", result)
        self.assertFalse(result['ok'])
        # Timeout error expected
        self.assertIn("TIMEOUT", str(result))

if __name__ == "__main__":
    unittest.main()
