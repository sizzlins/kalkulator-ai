import sys
import os
import subprocess

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.ai_utils import get_kimi

def debug_help():
    kimi = get_kimi()
    exe = kimi._find_executable()
    print(f"Executable: {exe}")
    
    print("\n--- Running -h ---")
    # Use errors='replace' to handle bad encoding
    try:
        res = subprocess.run([exe, "-h"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_help()
