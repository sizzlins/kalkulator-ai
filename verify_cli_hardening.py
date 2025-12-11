import sympy as sp
import re

# Mock Context
class MockContext:
    def __init__(self):
        self.variables = {}

ctx = MockContext()

def test_logic(raw_input, setup_vars=None):
    if setup_vars:
        ctx.variables = setup_vars
    else:
        ctx.variables = {}

    # Mimic split_top_level_commas (simplified for test)
    parts_check = [p.strip() for p in raw_input.split(",")]
    
    # Logic extracted from app.py
    is_function_finding = False
    
    if len(parts_check) >= 2 and all("=" in p for p in parts_check):
        func_names_found = set()
        numeric_args_count = 0
        valid_structure = True
        
        structure_pattern = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)\s*=\s*(.+)$")
        
        for p in parts_check:
            m = structure_pattern.match(p.strip())
            if not m:
                valid_structure = False
                break
                
            func_name = m.group(1)
            arg_str = m.group(2)
            
            func_names_found.add(func_name)
            
            # EVALUATION CHECK
            try:
                # Construct local dict from context
                local_vars = {name: info["result"] for name, info in ctx.variables.items()}
                
                if arg_str.strip() in ("pi", "e", "E"):
                    is_numeric = True
                elif arg_str.strip() in local_vars:
                    val = local_vars[arg_str.strip()]
                    try:
                        parsed_val = sp.sympify(val)
                        is_numeric = parsed_val.is_number
                    except:
                        is_numeric = False
                else:
                    try:
                        parsed_arg = sp.sympify(arg_str, locals={"pi": sp.pi, "e": sp.E})
                        free_syms = parsed_arg.free_symbols
                        # Check for unsafe symbols (not in context)
                        # context values are strings in real app, but here mock might differ?
                        # In real app ctx.variables['x']['result'] is string.
                        
                        unsafe_syms = [s for s in free_syms if str(s) not in local_vars and str(s) not in ("pi", "e", "E")]
                        
                        if not unsafe_syms:
                            is_numeric = True
                        else:
                            is_numeric = False
                    except Exception as e:
                        is_numeric = False
                        
                if is_numeric:
                    numeric_args_count += 1
            except:
                pass
                
        if valid_structure and len(func_names_found) == 1 and numeric_args_count >= 2:
             is_function_finding = True
             
    return is_function_finding

# Test Cases
print(f"1. f(pi)=3, f(0)=0 -> {test_logic('f(pi)=3, f(0)=0')}") # Expected: True
print(f"2. f(x)=2, f(y)=3 (x,y unknown) -> {test_logic('f(x)=2, f(y)=3')}") # Expected: False
print(f"3. f(3)=9, f(4)=16 -> {test_logic('f(3)=9, f(4)=16')}") # Expected: True

# Context Test: x=5
ctx_vars = {"x": {"result": "5"}}
print(f"4. f(x)=25, f(x+1)=36 (with x=5) -> {test_logic('f(x)=25, f(x+1)=36', ctx_vars)}") # Expected: True

# Context Test: x is symbolic in context? (Not possible in current variable system usually, but consistent)
print(f"5. g(a)=1, g(b)=2 (unknowns) -> {test_logic('g(a)=1, g(b)=2')}") # Expected: False
