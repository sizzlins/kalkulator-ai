"""Test cache hit flow - requires multiprocessing guard on Windows."""
import json


def main():
    """Main test function - must be in function for Windows multiprocessing."""
    from kalkulator_pkg.cache_manager import (
        clear_cache_hits,
        get_cache_hits,
        update_eval_cache,
        update_subexpr_cache,
    )
    from kalkulator_pkg.worker import evaluate_safely

    # Setup cache
    clear_cache_hits()
    result_dict = {"ok": True, "result": "8", "approx": "8"}
    update_eval_cache("4+4", json.dumps(result_dict))
    update_subexpr_cache("4+4", "8")

    print("=== Testing evaluate_safely with cached 4+4 ===")
    clear_cache_hits()
    result = evaluate_safely("4+4")
    print(f"Result OK: {result.get('ok')}")
    print(f"Result: {result.get('result')}")
    print(f"Cache hits in result: {result.get('cache_hits')}")
    print(f"Cache hits from get_cache_hits(): {get_cache_hits()}")

    print("\n=== Testing evaluate_safely with (4+4)+4 ===")
    clear_cache_hits()
    result2 = evaluate_safely("(4+4)+4")
    print(f"Result OK: {result2.get('ok')}")
    print(f"Result: {result2.get('result')}")
    print(f"Cache hits in result: {result2.get('cache_hits')}")
    print(f"Cache hits from get_cache_hits(): {get_cache_hits()}")

    print("\n=== All cache tests completed ===")

if __name__ == "__main__":
    # Required guard for Windows multiprocessing
    main()
