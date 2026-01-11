"""Test REPL integration of --transform flag."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

# Simulate REPL environment
from kalkulator_pkg.cli.repl_commands import _handle_evolve

# Test data for (1+x)^(1/x)
test_command = "evolve f(x) from x=[1,2,3,4], y=[2.0, 1.732, 1.587, 1.495]"

print("="*70)
print("Testing REPL --transform Integration")
print("="*70)
print()

# Test 1: Without --transform (old way)
print("Test 1: WITHOUT --transform (direct space only)")
print(f"Command: {test_command}")
print()
_handle_evolve(test_command, None)

print()
print("="*70)
print()

# Test 2: With --transform (new way)
transform_command = "evolve --transform --verbose f(x) from x=[1,2,3,4], y=[2.0, 1.732, 1.587, 1.495]"
print("Test 2: WITH --transform (multi-space)")
print(f"Command: {transform_command}")
print()
_handle_evolve(transform_command, None)
