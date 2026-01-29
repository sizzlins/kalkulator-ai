"""Test REPL cache commands."""
import subprocess
import sys

# Test showcache command via REPL
commands = """2+2
3*4
showcache
quit
"""

result = subprocess.run(
    [sys.executable, "kalkulator.py"],
    input=commands,
    capture_output=True,
    text=True,
    cwd=r"c:\Users\LOQ\PycharmProjects\kalkulator-ai",
    timeout=30
)

print("=== REPL Cache Command Test ===\n")
print("STDOUT:")
print(result.stdout)

if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

# Check if showcache output is present
if "Cache" in result.stdout or "cache" in result.stdout:
    print("\n✓ showcache command appears to work")
else:
    print("\n✗ showcache output not found")
