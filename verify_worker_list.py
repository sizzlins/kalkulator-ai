
import sys
import unittest
import numpy as np
import sympy as sp
from kalkulator_pkg.worker import worker_evaluate

class TestWorkerListEvaluation(unittest.TestCase):
    def test_list_evaluation(self):
        print("Testing worker evaluation of list...")
        # worker_evaluate expects an expression string or potentially pre-parsed object if internal?
        # Looking at worker.py line 383, it receives `expr`.
        # `worker_evaluate` definition: def worker_evaluate(preprocessed_expr: str, ...):
        # But wait, it calls `parse_preprocessed` first.
        # If we want to test the crash at logic line 383: `res = sp.simplify(expr)`
        # We need `expr` to be a list at that point.
        # This means `parse_preprocessed` must return a list.
        # And we enabled that in parser.py.
        
        # So we can pass a string representation of a list to worker_evaluate?
        # No, `worker_evaluate` is designed to run in a separate process often, but `evaluate_safely` handles that.
        # Let's import `worker_evaluate` and call it directly with a list object effectively mocking the parse result?
        # No, `worker_evaluate` takes a string and calls parse.
        # Let's bypass `worker_evaluate` and test the function that contains the logic if possible, 
        # or mock `parse_preprocessed` to return a list.
        
        # Actually, let's just use `evaluate_safely` from worker (which handles the full pipeline if we use process? No).
        # We can call `worker_evaluate` with a string "[1, 2, 3]".
        # CAUTIION: `worker_evaluate` is the function that crashed.
        
        try:
             # We need to simulate the flow where `parse_preprocessed` returns a list.
             # We can't easily mock inside the test without patching. 
             # But if we pass "[1, 2, 3]" string, `parse_preprocessed` should return a list now (since we fixed parser).
             # So calling `worker_evaluate("[1, 2, 3]")` should trigger the code path.
             
             result = worker_evaluate("[1, 2, 3]")
             print(f"Result: {result}")
             
             if not result['ok']:
                 print(f"FAIL: Worker returned error: {result.get('error')}")
                 # If error is about 'replace', we failed.
                 if "object has no attribute 'replace'" in str(result.get('error')):
                     raise AttributeError("Still crashing on simplify")
             else:
                 print("PASS: Worker handled list without crash.")
                 
        except Exception as e:
            print(f"CRASH: {e}")
            raise e

if __name__ == "__main__":
    unittest.main()
