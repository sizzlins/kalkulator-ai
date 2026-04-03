import kalkulator_pkg.config as config
import sympy as sp

print(f"Loading config...")
try:
    allowed = config.ALLOWED_SYMPY_NAMES
    print(f"Allowed names loaded. Count: {len(allowed)}")
    
    if "max" in allowed:
        print(f"'max' found! Value: {allowed['max']}")
        print(f"Type: {type(allowed['max'])}")
        print(f"Is sp.Max? {allowed['max'] is sp.Max}")
    else:
        print("'max' NOT found in config.ALLOWED_SYMPY_NAMES")
        
    if "min" in allowed:
        print(f"'min' found! Value: {allowed['min']}")
    else:
        print("'min' NOT found")

    # Double check delegation to sympy_defs
    import kalkulator_pkg.sympy_defs as defs
    print(f"\nChecking sympy_defs directly:")
    if "max" in defs.ALLOWED_SYMPY_NAMES:
         print(f"'max' in sympy_defs: {defs.ALLOWED_SYMPY_NAMES['max']}")
    else:
         print("'max' NOT in sympy_defs")
         
except Exception as e:
    print(f"Error: {e}")
