
import os

target_file = r"C:\Users\LOQ\PycharmProjects\kalkulator-ai\kalkulator_pkg\symbolic_regression\genetic_config.py"

with open(target_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
found = False

for line in lines:
    new_lines.append(line)
    # Search for any line defining random_state, ignoring exact whitespace
    if not found and "random_state" in line and ":" in line and "int" in line:
        # Add new fields
        new_lines.append("\n")
        new_lines.append("    # Loss Function (Robust Regression)\n")
        new_lines.append("    loss_function: str = \"mse\" # \"mse\" or \"huber\"\n")
        new_lines.append("    huber_delta: float = 1.35\n")
        found = True

if found:
    print("Found and updated GeneticConfig.")
    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
else:
    print("Could not find insertion point in GeneticConfig.")
