
import os
import sys
import builtins
from unittest.mock import MagicMock

# 1. Mock input to return a command
input_mock = MagicMock(side_effect=["162 + 953", EOFError])
builtins.input = input_mock

# 2. Clear cache first
from kalkulator_pkg.cache_manager import clear_persistent_cache
clear_persistent_cache()

# 3. Import REPL
from kalkulator_pkg.cli.repl_core import REPL

print("Initializing REPL...")
repl = REPL()

# 4. Run loop_once (Simulate "162 + 953")
# This should:
# - Process "162 + 953" -> Eval -> "1115"
# - Update in-memory cache
# - Call save_cache_to_disk() (THE FIX)
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
