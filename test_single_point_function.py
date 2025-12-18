#!/usr/bin/env python3
"""Test single data point function finding like f(-1)=3"""

# Simulate what happens when user types f(-1)=3
import re

from kalkulator_pkg.function_manager import (
    find_function_from_data,
    parse_find_function_command,
)

# Test the pattern matching
raw = "f(-1)=3"
func_assignment_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]+\)\s*=\s*[^,]+'
matches = list(re.finditer(func_assignment_pattern, raw))
print(f"Pattern matches for '{raw}': {len(matches)}")
for m in matches:
    print(f"  Match: {m.group(0)}")

# Test function finding with single point
if len(matches) >= 1:
    func_name = matches[0].group(1)
    print(f"\nFunction name: {func_name}")
    
    # Extract arguments
    args_match = re.search(rf'{re.escape(func_name)}\s*\(([^)]+)\)', matches[0].group(0))
    if args_match:
        args_str = args_match.group(1)
        print(f"Arguments: {args_str}")
        
        # For single parameter, use 'x'
        param_names = ['x']
        param_str = ", ".join(param_names)
        raw_with_find = f"{raw}, find {func_name}({param_str})"
        print(f"Modified input: {raw_with_find}")
        
        # Parse as find command
        find_func_cmd = parse_find_function_command(raw_with_find)
        if find_func_cmd:
            print(f"Find command parsed: {find_func_cmd}")
            
            # Extract data point
            from kalkulator_pkg.parser import split_top_level_commas
            find_pattern = rf"find\s+{re.escape(func_name)}\s*\([^)]*\)"
            data_str = re.sub(find_pattern, "", raw, flags=re.IGNORECASE).strip()
            data_str = data_str.rstrip(',').strip()
            print(f"Data string: {data_str}")
            
            # Parse data point
            parts = split_top_level_commas(data_str)
            data_points = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                pattern = rf"{re.escape(func_name)}\s*\(([^)]+)\)\s*=\s*(.+)"
                match = re.match(pattern, part)
                if match:
                    args_str = match.group(1)
                    value_str = match.group(2).strip()
                    print(f"  Found data point: args={args_str}, value={value_str}")
                    
                    # Parse arguments
                    arg_list = split_top_level_commas(args_str)
                    final_args = []
                    for arg in arg_list:
                        try:
                            arg_val = float(arg.strip())
                            final_args.append(arg_val)
                        except ValueError:
                            final_args.append(arg.strip())
                    
                    # Parse value
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str
                    
                    data_points.append((final_args, value))
            
            print(f"\nData points: {data_points}")
            
            # Find function
            success, func_str, factored_form, error_msg = find_function_from_data(
                data_points, param_names
            )
            
            if success:
                print(f"\n[SUCCESS] Function found: f(x) = {func_str}")
                if factored_form:
                    print(f"  Equivalent: f(x) = {factored_form}")
            else:
                print(f"\n[FAILED] {error_msg}")

