
import os

file_path = "kalkulator_pkg/cli/repl_commands.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "AltCall" in line:
            print(f"Line {i+1}: {line.strip()}")
            # Print context
            for k in range(max(0, i-5), min(len(lines), i+10)):
                print(f"  {k+1}: {lines[k].rstrip()}")
            break
