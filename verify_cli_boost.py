
from kalkulator_pkg.cli.repl_commands import _handle_evolve
import sys
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

try:
    # Test valid boost
    cmd = "evolve f(x) from x=[1,2,3], y=[1,2,3] --boost 2"
    _handle_evolve(cmd)
    output = mystdout.getvalue()
    if "Boosting Round" in output or "Starting evolution" in output: # Verbose might show boosting or evolution
        print("Success: CLI accepted boost flag.")
    else:
        # It might run too fast or verbose off? Config default is verbose=True in repl_commands
        print(f"Check Output:\n{output}")

except Exception as e:
    print(f"CLI Error: {e}")
finally:
    sys.stdout = old_stdout
