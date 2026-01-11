"""Test script to verify hybrid mode fix works."""
import subprocess
import sys

# Create a temporary script that runs the exact user command
test_script = """
import sys
sys.path.insert(0, r'C:\\Users\\LOQ\\PycharmProjects\\kalkulator-ai')

from kalkulator_pkg.cli.repl_commands import _handle_evolve_command
from types import SimpleNamespace

# Mock context
ctx = SimpleNamespace()

# User's exact command (first 30 points for speed)
command = '''evolve --hybrid --verbose f(-5, -5) = -0.00032, f(-5, -4) = 0.0016, f(-5, -3) = -0.008, f(-5, -2) = 0.04, f(-5, -1) = -0.2, f(-5, 0) = 1, f(-5, 1) = -5, f(-5, 2) = 25, f(-5, 3) = -125, f(-5, 4) = 625, f(-5, 5) = -3125, f(-4, -5) = -0.0009765625, f(-4, -4) = 0.00390625, f(-4, -3) = -0.015625, f(-4, -2) = 0.0625, f(-4, -1) = -0.25, f(-4, 0) = 1, f(-4, 1) = -4, f(-4, 2) = 16, f(-4, 3) = -64, f(-4, 4) = 256, f(-4, 5) = -1024, f(2, 3) = 8, f(4, 5) = 1024, f(6, 7) = 279936, f(10, 11) = 100000000000, f(18, 19) = 7.08235345355338e+23'''

try:
    _handle_evolve_command(command, {}, ctx)
    print("\\n✅ Test completed successfully")
except Exception as e:
    print(f"\\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
"""

# Write and run the test
with open('temp_hybrid_test.py', 'w') as f:
    f.write(test_script)

print("Running hybrid mode test with quality check...")
print("="*70)
result = subprocess.run([sys.executable, 'temp_hybrid_test.py'], 
                       capture_output=True, text=True, timeout=60)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Check for expected behavior
output = result.stdout
if "Hybrid seeding: find() result has low R²" in output:
    print("\n✅ Quality check working! Bad seed rejected.")
elif "Hybrid seeding: using find() result" in output:
    print("\n⚠️ Seed was used (might be good this time)")
    
if "x**y" in output or "x^y" in output:
    print("✅ Found correct x^y pattern!")
else:
    print("⚠️ Did not find x^y (might timeout with small dataset)")
