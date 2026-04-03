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
    
    print("\n--- Running --help ---")
    res = subprocess.run([exe, "--help"], capture_output=True, text=True, encoding='utf-8')
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    print("\n--- Running help ---")
    res = subprocess.run([exe, "help"], capture_output=True, text=True, encoding='utf-8')
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

if __name__ == "__main__":
    debug_help()
