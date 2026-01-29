
import sys
import os
import json
import time

# Ensure importable
sys.path.insert(0, os.getcwd())

from kalkulator_pkg.worker import _WORKER_MANAGER, warmup_workers, evaluate_safely

def verify_simple_ipc():
    print("Testing simple IPC (1+1)...")
    try:
        # 1. Warmup (starts workers)
        warmup_workers()
        print("Warmup calls completed (or ignored).")
        
        # 2. Check is_alive
        if _WORKER_MANAGER.is_alive():
            print(f"Worker Manger is alive. Procs: {len(_WORKER_MANAGER.procs)}")
        else:
            print("Worker Manager is DEAD.")
            _WORKER_MANAGER.start()
            print("Restarted Worker Manager.")

        # 3. Direct Request
        print("Sending direct request...")
        resp = _WORKER_MANAGER.request(
             {"type": "eval", "preprocessed": "1+1"}, timeout=5
        )
        print(f"Direct Response: {resp}")
        
        if resp and resp.get("ok"):
            print("SUCCESS: IPC Works.")
        else:
            print("FAIL: IPC response invalid or None.")

    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_simple_ipc()
