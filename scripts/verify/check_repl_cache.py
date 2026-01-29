
from kalkulator_pkg.cache_manager import get_cached_eval, load_persistent_cache

# Force load from disk
cache = load_persistent_cache()

# Need to know the preprocessed key. "162 + 953" usually becomes "162 + 953" (normalized spaces).
# But wait, parser.preprocess might change it.
# Let's check "162 + 953" and variations.

# Better: search values
found = False
for k, v in cache.get("eval_cache", {}).items():
    res = v.get("result")
    if res == "1115" or res == "1115.0":
        found = True
        print(f"SUCCESS: Found cached item: {k} -> {res}")
        break

if not found:
    print("FAILURE: Cache item for '1115' missing (Data Loss Confirmed).")
    print(f"Cache content count: {len(cache.get('eval_cache', {}))}")
