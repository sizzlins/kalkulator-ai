
from kalkulator_pkg.cache_manager import get_cached_eval, load_persistent_cache

# Force load from disk
cache = load_persistent_cache()
entry = get_cached_eval("crash_test_key")

if entry:
    print(f"SUCCESS: Found cached item: {entry}")
else:
    print("FAILURE: Cache item missing (Data Loss Confirmed).")
