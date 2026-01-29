
import os

target_file = r"c:\Users\LOQ\PycharmProjects\kalkulator-ai\kalkulator_pkg\symbolic_regression\genetic_engine.py"

with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

new_block = """        default_factory=lambda: {
            "max": 5.0,  # Tier 3: Penalize Piecewise Cheating
            "min": 5.0,  # Tier 3
            "abs": 4.0,  # Tier 3: The "Gateway Drug" to max() - heavily penalized
            
            # Tier 2: Physics (Subsidized to match fundamental cost)
            "sin": 1.0,
            "cos": 1.0,
            "tan": 1.0,
            "asin": 1.0,
            "acos": 1.0,
            "atan": 1.0,
            "exp": 1.0,
            "log": 1.0,
            "plog": 1.0,
            "sqrt": 1.0,
            "psqrt": 1.0,
            "pow": 1.0,  # Make 2^x as cheap as 2*x
            
            # Tier 1: Fundamental
            "add": 1.0,
            "sub": 1.0,
            "mul": 1.0,
            "div": 1.0,
        }"""

start_marker = 'default_factory=lambda: {'
end_marker = '        }'
    
start_idx = content.find(start_marker)
if start_idx != -1:
     # Find the closing brace after start
     # We need to be careful not to find a nested brace, but this dict is flat.
     end_idx = content.find(end_marker, start_idx)
     if end_idx != -1:
         print(f"Found block from {start_idx} to {end_idx}. Replacing...")
         
         # The range to replace is from start_idx to end_idx + len(end_marker)
         # new_block includes the markers.
         
         # Ensure new_block leading whitespace matches file? 
         # The file seems to use 8 spaces indentation for default_factory.
         # My new_block has 8 spaces.
         
         final_content = content[:start_idx] + new_block + content[end_idx+len(end_marker):]
         
         with open(target_file, "w", encoding="utf-8") as f:
            f.write(final_content)
         print("SUCCESS: File updated and saved.")
     else:
         print("FAIL: Could not find end marker")
else:
     print("FAIL: Could not find start marker")
