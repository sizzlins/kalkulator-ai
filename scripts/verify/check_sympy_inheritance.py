import sympy as sp
import inspect
from kalkulator_pkg.sympy_defs import ALLOWED_SYMPY_NAMES

print(f"Auditing {len(ALLOWED_SYMPY_NAMES)} allowed function names for inheritance issues...")

non_functions = []
safe_types = (sp.Integer, sp.Float, sp.Rational)

# Already whitelisted in parser.py
explicit_whitelist = (sp.Max, sp.Min, sp.Piecewise)

for name, obj in ALLOWED_SYMPY_NAMES.items():
    # Only care about classes (types), not instances or functions
    if not isinstance(obj, type):
        continue
        
    # Check if it's a subclass of sp.Function
    is_f = issubclass(obj, sp.Function)
    
    if not is_f:
        # Check against parser.py logic
        is_safe_num = issubclass(obj, safe_types)
        is_explicit = obj in explicit_whitelist
        
        status = "❌ REJECTED"
        if is_safe_num: status = "✅ SAFE (Number)"
        if is_explicit: status = "✅ SAFE (Explicit)"
        
        # Internal SymPy classes (Expr, Basic) are explicitly handled (rejected/symbolized) 
        # but we want to know about things like 'Mod', 'Abs', etc.
        
        print(f"[{status}] '{name}' -> {obj.__name__} (Bases: {[b.__name__ for b in obj.__bases__]})")
        
        if status.startswith("❌"):
            non_functions.append(name)

print("\nSummary of potential issues:")
print(non_functions)
