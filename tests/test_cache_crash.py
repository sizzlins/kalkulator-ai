
import os
import sys
import time

# Mock the REPL adding an item
from kalkulator_pkg.cache_manager import clear_persistent_cache, update_eval_cache, load_persistent_cache

# 1. Clear first
clear_persistent_cache()
print("Cache cleared.")

# 2. Add item (In-memory update)
# This simulates what happens when worker returns result and we update cache
update_eval_cache("crash_test_key", '{"result": "survived"}')
print("Item added to memory.")

# 3. Simulate hard exit (like SIGKILL or closing terminal window)
# This bypasses any 'finally' blocks in main()
print("Simulating hard crash (os._exit)...")
sys.stdout.flush()
os._exit(0)
