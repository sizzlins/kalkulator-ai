import sympy as sp
import numpy as np

def eval_to_float(val):
    """
    Convert a value (string or number) to float, evaluating symbolic expressions.
    Handles 'pi', 'e', 'inf', 'nan', etc.
    """
    if isinstance(val, (int, float)):
        return float(val)

    # Handle SymPy infinity types BEFORE general conversion
    # zoo = ComplexInfinity, oo = positive infinity
    if val is sp.zoo or val is sp.oo or val is sp.S.Infinity:
        return float("inf")
    if val is sp.S.NegativeInfinity:
        return float("-inf")
    if val is sp.nan or val is sp.S.NaN:
        return float("nan")

    if isinstance(val, str):
        # Check for string representations of infinity
        val_lower = val.lower().strip()
        if val_lower in ("zoo", "oo", "inf", "infinity", "complexinfinity"):
            return float("inf")
        if val_lower in ("nan", "-nan"):
            return float("nan")

        try:
            # Try direct conversion first
            return float(val)
        except ValueError:
            pass
            
        # Try evaluating as a symbolic expression
        try:
            local_ns = {
                "pi": sp.pi,
                "e": sp.E,
                "E": sp.E,
                "I": sp.I,
                "i": sp.I,  # Add 'i' for complex numbers
                "sqrt": sp.sqrt,
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "log": sp.log,
                "ln": sp.log,
                "exp": sp.exp,
                "zoo": sp.zoo,
                "oo": sp.oo,
            }
            expr = sp.sympify(val, locals=local_ns)

            # Check if result is infinity type
            if expr is sp.zoo or expr is sp.oo or expr is sp.S.Infinity:
                return float("inf")
            if expr is sp.S.NegativeInfinity:
                return float("-inf")
            if expr is sp.nan or expr is sp.S.NaN:
                return float("nan")
                
            # Convert to complex or float
            res = complex(expr)
            if res.imag == 0:
                return res.real
            return res # Return complex if it is complex (e.g. 1+2j)
            
        except Exception as e:
            # Last ditch: simple python eval (risky but restricted?)
            # No, avoid eval() for safety unless sanitized.
            # sympify usually handles most math.
            raise ValueError(f"Could not convert '{val}' to float/complex: {e}")
            
    # Try converting other types
    try:
        return float(val)
    except Exception as e:
        raise ValueError(f"Could not convert {type(val)} to float: {e}")
