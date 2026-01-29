
import os

target_file = r"c:\Users\LOQ\PycharmProjects\kalkulator-ai\kalkulator_pkg\function_finder_advanced.py"

def patch_file():
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Range to replace: 1161 to 1508 (1-based)
    # Indices: 1160 to 1508 (exclusive in slice logic? No, 1508 lines means index 1507 is line 1508)
    # So we want to remove lines[1160] up to lines[1507] inclusive.
    # Python slice [start:end] excludes end. So [1160:1508].
    
    start_idx = 1160
    end_idx = 1508
    
    print(f"Replacing lines {start_idx+1} to {end_idx}")
    print(f"First line to remove: {lines[start_idx].strip()}")
    print(f"Last line to remove: {lines[end_idx-1].strip()}")
    
    # Prepare replacement
    replacement = [
        "            # Refactored Transcendental Features logic\n",
        "            _add_transcendental_features(col, name, features, feature_names, y_data)\n"
    ]
    
    new_lines = lines[:start_idx] + replacement + lines[end_idx:]
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("Patch applied.")

if __name__ == "__main__":
    patch_file()
