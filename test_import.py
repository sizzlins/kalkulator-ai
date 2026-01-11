import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from kalkulator_pkg.cli import repl_commands
    print("Import OK")
except Exception as e:
    import traceback
    traceback.print_exc()
