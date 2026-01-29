"""Test script for cache system verification."""
from kalkulator_pkg import cache_manager as cm

print("=== Cache System Test ===\n")

# 1. Get cache
cache = cm.get_persistent_cache()
print(f"1. Cache loaded: {type(cache).__name__}")
print(f"   Keys: {list(cache.keys())}")

# 2. Check eval cache size
eval_cache = cache.get("eval_cache", {})
print(f"2. Eval cache entries: {len(eval_cache)}")

# 3. Check subexpr cache size  
subexpr_cache = cache.get("subexpr_cache", {})
print(f"3. Subexpr cache entries: {len(subexpr_cache)}")

# 4. Test update and retrieve
test_key = "test_cache_verification_expr"
test_value = '{"result": 42}'
cm.update_eval_cache(test_key, test_value, compute_time=0.001)
result = cm.get_cached_eval(test_key)
status4 = "PASS" if result == test_value else "FAIL"
print(f"4. Eval cache set/get: {status4}")
if result != test_value:
    print(f"   Expected: {test_value}")
    print(f"   Got: {result}")

# 5. Test subexpr cache
cm.update_subexpr_cache("test_2+2", "4", compute_time=0.0001)
subresult = cm.get_cached_subexpr("test_2+2")
status5 = "PASS" if subresult == "4" else "FAIL"
print(f"5. Subexpr cache set/get: {status5}")

# 6. Test timing retrieval
result_with_time, compute_time = cm.get_cached_eval_with_time(test_key)
status6 = "PASS" if compute_time is not None else "FAIL"
print(f"6. Timing retrieval: {status6} (time={compute_time})")

# 7. Check cache hits tracking
hits = cm.get_cache_hits()
print(f"7. Cache hits tracked: {len(hits)} entries")

# 8. Save to disk
try:
    cm.save_cache_to_disk()
    print("8. Save to disk: PASS")
except Exception as e:
    print(f"8. Save to disk: FAIL ({e})")

# 9. Clear and verify
cm.clear_cache_hits()
hits_after = cm.get_cache_hits()
status9 = "PASS" if len(hits_after) == 0 else "FAIL"
print(f"9. Clear cache hits: {status9}")

# 10. Test persistence - reload cache
cache2 = cm.load_persistent_cache()
eval_cache2 = cache2.get("eval_cache", {})
status10 = "PASS" if test_key in eval_cache2 else "FAIL"
print(f"10. Persistence check: {status10}")

print("\n=== Cache System Test Complete ===")

# Summary
all_passed = all(s == "PASS" for s in [status4, status5, status6, status9, status10])
if all_passed:
    print("All tests PASSED!")
else:
    print("Some tests FAILED - review output above")
