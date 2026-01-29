
import sys
import traceback

print(f"Python: {sys.version}")
print(f"Path: {sys.path}")

try:
    import kalkulator_pkg.parser as p
    print("Successfully imported parser.")
    print("Attributes in parser:")
    print([x for x in dir(p) if 'preprocess' in x])
    
    if hasattr(p, 'preprocess_expression'):
        print("preprocess_expression FOUND.")
    else:
        print("preprocess_expression NOT FOUND.")
        
except Exception:
    traceback.print_exc()
