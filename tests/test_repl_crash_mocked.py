
import os
import sys
import builtins
from unittest.mock import MagicMock, patch

# 1. Clear cache first
from kalkulator_pkg.cache_manager import clear_persistent_cache, update_eval_cache
clear_persistent_cache()

# 2. Key for testing
TEST_EXPR = "162 + 953"
TEST_RES = "1115"

# 3. Setup Mocks
# Mock input
input_mock = MagicMock(side_effect=[TEST_EXPR, EOFError])
builtins.input = input_mock

# Mock evaluate_safely to simulate worker returning result AND updating memory cache
# We must patch where it is IMPORTED in repl_core, or patch the definition.
# repl_core imports it: 'from ..worker import evaluate_safely' inside methods usually.
# But verify import style. Lines 729: 'from ..worker import evaluate_safely'.
# So we must patch 'kalkulator_pkg.worker.evaluate_safely'.

def mock_evaluate_safely(expr, **kwargs):
    # Simulate worker updating cache
    print(f"Mock Worker: Evaluating {expr} and updating cache...")
    update_eval_cache(expr, f'{{"result": "{TEST_RES}"}}')
    return {"ok": True, "result": TEST_RES}

# 4. Run REPL with patch
with patch('kalkulator_pkg.worker.evaluate_safely', side_effect=mock_evaluate_safely):
    from kalkulator_pkg.cli.repl_core import REPL
    print("Initializing REPL...")
    repl = REPL()

    print("Running loop_once...")
    try:
        repl.loop_once()
    except SystemExit:
        pass
    except EOFError:
        pass

print("Command processed.")

# 5. Simulate hard crash
print("Simulating hard crash...")
sys.stdout.flush()
os._exit(0)
