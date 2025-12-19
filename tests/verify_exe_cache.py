import os
import subprocess

# Assume dist/kalkulator.exe
EXE_PATH = os.path.join(os.getcwd(), "dist", "kalkulator.exe")


def run_interaction(commands):
    """Run the exe, send commands, and return full output."""
    if not os.path.exists(EXE_PATH):
        raise FileNotFoundError(f"EXE not found at {EXE_PATH}")

    process = subprocess.Popen(
        [EXE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,  # Unbuffered
    )

    try:
        input_str = "\n".join(commands) + "\n"
        stdout, stderr = process.communicate(input=input_str, timeout=10)
        return stdout + stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return "TIMEOUT"
    except Exception as e:
        process.kill()
        return str(e)


def test_cache_persistence():
    print(f"Verifying Cache Persistence: {EXE_PATH}")

    # 1. Clear Cache
    print("Step 1: Clearing cache...")
    out1 = run_interaction(["clearcache", "quit"])
    if "Cache cleared" in out1:
        print("   [OK] Cache cleared command executed.")
    else:
        print("   [WARN] Cache clear output not standard (expected 'Cache cleared').")

    # 2. Perform Calculation
    print("Step 2: Performing calculation (should populate cache)...")
    expr = "factorial(20)"
    out2 = run_interaction([expr, "quit"])

    # We'll just look for a result that looks like a number

    # factorial(20) = 2432902008176640000
    expected_part = "2432902008176640000"

    if expected_part in out2:
        print(f"   [OK] Calculation correct: found {expected_part}")
    else:
        print("   [FAIL] Calculation incorrect or not found.")
        print(out2)
        # return # Try checking cache anyway

    # 3. Check Cache (New Session)
    print("Step 3: Checking cache in a new session...")
    out3 = run_interaction(["showcache", "quit"])

    # Check if the result appears in the cache dump
    if expected_part in out3:
        print("   [OK] Persistent cache entry found (value match).")
    elif "factorial(20)" in out3:
        print("   [OK] Persistent cache entry found (key match).")
    elif "Entries: 1" in out3 or "1 entries" in out3:
        print("   [OK] Cache entry count confirms persistence.")
    else:
        print("   [FAIL] Cache does not seem to persist.")
        print("   Output dump from showcache:")
        print(out3)


if __name__ == "__main__":
    if not os.path.exists("dist"):
        os.makedirs("dist")
    test_cache_persistence()
