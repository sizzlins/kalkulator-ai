"""Quick test of alt shortcut."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

from kalkulator_pkg.cli.repl_core import ReplCore

repl = ReplCore()

print("Testing 'alt' shortcut...")
print()

# Simulate the command
repl.process_input("alt f(x) from x=[1,2,3], y=[2.0, 1.732, 1.587]")
