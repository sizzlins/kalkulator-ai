
import numpy as np
import re

def parse_function_data(arg):
    # Copied logic from repl_commands.py (simplified)
    # >>> alt f(-5)=0...
    
    # 1. Split arg
    arg = arg.strip()
    
    # regex for f(val)=val
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)\s*=\s*([^,]+)")
    
    points = []
    func_name = None
    
    # Split by comma respecting parens?
    # Simple split by comma for now as per user input
    parts = arg.split(',')
    
    X_list = []
    y_list = []
    
    for part in parts:
        match = pattern.search(part)
        if match:
            fname = match.group(1)
            x_val = match.group(2)
            y_val = match.group(3)
            
            if func_name is None:
                func_name = fname
                
            try:
                # safe_float simulation
                x_f = float(x_val)
                y_f = float(y_val)
                X_list.append(x_f)
                y_list.append(y_f)
            except:
                pass
                
    X = np.array(X_list).reshape(-1, 1)
    y = np.array(y_list)
    return X, y

def reproduce_via_mock():
    print("Reproducing via MOCK parsing...")
    input_str = "f(-5)=0, f(-4)=0, f(-3)=0, f(-2)=0, f(-1)=0, f(0)=0, f(1)=1, f(2)=2, f(3)=3, f(4)=4, f(5)=5"
    
    X, y = parse_function_data(input_str)
    
    print(f"Parsed X shape: {X.shape}")
    print(f"X sample: {X.flatten()}")
    
    if np.any(X < 0):
        print("X HAS NEGATIVE VALUES.")
    else:
        print("X HAS NO NEGATIVE VALUES.")

if __name__ == "__main__":
    reproduce_via_mock()
