"""Debug test for symbolic constants in function finding."""
import sys

sys.path.insert(0, ".")

if __name__ == "__main__":
    # Manually call eval_to_float with debug
    import sympy as sp
    
    def eval_to_float(val):
        """Convert a value (string or number) to float, evaluating symbolic expressions."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                # Try direct conversion first
                return float(val)
            except ValueError:
                pass
            # Try evaluating as a symbolic expression
            try:
                local_ns = {
                    'pi': sp.pi, 'e': sp.E, 'E': sp.E, 'I': sp.I,
                    'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                    'log': sp.log, 'ln': sp.log, 'exp': sp.exp
                }
                expr = sp.sympify(val, locals=local_ns)
                # Force numeric evaluation
                result = expr.evalf()
                # Check if result is still symbolic (contains free symbols)
                if hasattr(result, 'free_symbols') and result.free_symbols:
                    raise ValueError(f"Expression '{val}' contains free symbols: {result.free_symbols}")
                return float(result)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Could not convert '{val}' to a numeric value: {e}")
            except Exception as e:
                raise ValueError(f"Could not convert '{val}' to a numeric value")
        # For SymPy expressions
        try:
            result = val.evalf()
            if hasattr(result, 'free_symbols') and result.free_symbols:
                raise ValueError(f"Expression contains free symbols: {result.free_symbols}")
            return float(result)
        except Exception:
            raise ValueError(f"Could not convert '{val}' to a numeric value")

    
    # Test data points
    data_points = [
        (["0"], "0"),
        (["pi/2"], "pi/2"),
        (["pi"], "0"),
        (["4.71"], "-4.71"),
    ]
    
    print("Testing eval_to_float on data points:")
    for x_tuple, y_val in data_points:
        x_val = x_tuple[0]
        try:
            x_float = eval_to_float(x_val)
            y_float = eval_to_float(y_val)
            print(f"  x='{x_val}' -> {x_float}, y='{y_val}' -> {y_float}")
        except Exception as e:
            print(f"  ERROR on x='{x_val}', y='{y_val}': {e}")
    
    # Now test using find_function_from_data directly
    print("\nTesting find_function_from_data:")
    from kalkulator_pkg.function_manager import find_function_from_data
    
    try:
        success, func_str, factored, error = find_function_from_data(data_points, ["x"])
        if success:
            print(f"SUCCESS: g(x) = {func_str}")
        else:
            print(f"FAILURE: {error}")
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
