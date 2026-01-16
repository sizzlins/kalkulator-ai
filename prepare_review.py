
import os

FILES_TO_INCLUDE = [
    "kalkulator_pkg/symbolic_regression/genetic_engine.py",
    "kalkulator_pkg/symbolic_regression/genetic_config.py",  # NEW
    "kalkulator_pkg/symbolic_regression/strategies.py",      # NEW
    "kalkulator_pkg/symbolic_regression/forensic_analysis.py",
    "kalkulator_pkg/symbolic_regression/expression_tree.py",
    "kalkulator_pkg/symbolic_regression/operators.py",
    "kalkulator_pkg/parser.py",
    "kalkulator_pkg/tokenizer.py",                           # NEW
    "kalkulator_pkg/worker.py",
    "kalkulator_pkg/cli/repl_commands.py",
    "kalkulator_pkg/config.py",
    "kalkulator_pkg/solver/dispatch.py",
]

OUTPUT_FILE = "GEMINI_REVIEW_PACKET.txt"

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # Write Tree Structure
        outfile.write("# PROJECT STRUCTURE\n")
        outfile.write("```\n")
        # Simulating tree output (simplified)
        for root, dirs, files in os.walk("kalkulator_pkg"):
            if "__pycache__" in root: continue
            level = root.replace("kalkulator_pkg", "").count(os.sep)
            indent = " " * 4 * (level)
            outfile.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if f.endswith(".py"):
                    outfile.write(f"{subindent}{f}\n")
        outfile.write("```\n\n")

        # Write File Contents
        for filepath in FILES_TO_INCLUDE:
            if os.path.exists(filepath):
                outfile.write(f"\n\n# FILE: {filepath}\n")
                outfile.write("="*80 + "\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")
                outfile.write("\n" + "="*80 + "\n")
            else:
                outfile.write(f"\n\n# FILE: {filepath} (NOT FOUND)\n")

    print(f"Successfully generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
