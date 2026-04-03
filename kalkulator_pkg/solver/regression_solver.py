
from typing import Any, List, Tuple, Optional, Dict
import numpy as np
from fractions import Fraction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Helper for consistent power formatting
def _format_power(base_name: str, exponent: float) -> str:
    if abs(exponent - round(exponent)) < 1e-8:
        return f"{base_name}^{int(round(exponent))}"
    # prefer decimal representation for non-integer exponents
    s = f"{exponent:.10g}"
    # remove trailing zeros
    s = s.rstrip("0").rstrip(".")
    return f"{base_name}^{s}"

# Helper for rational polishing (Common Sense simplification)
def polish_rational(val: float, tolerance: float = 2e-5, max_den: int = 2000) -> Any:
    try:
        f = Fraction(val).limit_denominator(max_den)
        # If denominator is huge, it's not simpler.
        if f.denominator > 1000 and max_den > 1000:
            pass
        
        if abs(float(f) - val) <= tolerance:
            return f
    except Exception:
        pass
    return val

def solve(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False,
    skip_linear: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Main entry point for regression-based solving.
    Tries Linear, then Polynomial/Rational checks.
    """
    # 1. Linear Regression
    if not skip_linear:
        res = solve_linear(data_points, param_names, verbose)
        if res[0]:
            return res

    # 2. Polynomial Regression (Degree 2)
    # This replaces the missing "Priority 1" loop with a standard poly fit
    res_poly = solve_polynomial(data_points, param_names, degree=2, verbose=verbose)
    if res_poly[0]:
        return res_poly
        
    return (False, None, None, "Regression failed to find a fit.")

def solve_linear(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    verbose: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    try:
        # Extract X and y (assuming numeric data structure from FinderDispatch)
        # numeric_data is list[tuple[list[float], float]]
        # But here we might receive raw data or mixed. 
        # FinderDispatch passes mixed data to Strategies, but Strategies merge them.
        # Let's assume input is list[tuple[list[float], float]] for simplicity
        # or handle conversion.
        
        X_vals = []
        y_vals = []
        
        for p in data_points:
            # p[0] is X (list of floats), p[1] is y (float)
            # handle if p[0] is raw tuple
            x_in = p[0]
            if isinstance(x_in, (list, tuple, np.ndarray)):
                row = [float(xi) for xi in x_in]
            else:
                row = [float(x_in)]
            X_vals.append(row)
            y_vals.append(float(p[1]))

        X_arr = np.array(X_vals)
        y_arr = np.array(y_vals)

        if len(X_vals) < 2:
            return (False, None, None, "Not enough data for linear regression")

        lr = LinearRegression()
        lr.fit(X_arr, y_arr)

        y_pred = lr.predict(X_arr)
        mse = np.mean((y_arr - y_pred) ** 2)

        if mse < 1e-9:
            parts = []
            for idx, coef in enumerate(lr.coef_):
                if abs(coef) < 1e-10:
                    continue
                
                # Rational Polish
                polished = polish_rational(coef)
                var_name = param_names[idx] if idx < len(param_names) else f"x{idx}"
                
                term = ""
                if isinstance(polished, Fraction) and polished.denominator != 1:
                    term = f"{polished}*{var_name}"
                else:
                    val = float(polished)
                    coef_round = round(val)
                    if abs(val - coef_round) < 1e-9:
                         if coef_round == 1: term = var_name
                         elif coef_round == -1: term = f"-{var_name}"
                         else: term = f"{int(coef_round)}*{var_name}"
                    else:
                        term = f"{val:.10g}*{var_name}"
                
                if not parts:
                    parts.append(term)
                else:
                    if term.startswith("-"):
                        parts.append(f"- {term[1:]}")
                    else:
                        parts.append(f"+ {term}")

            intercept = lr.intercept_
            if abs(intercept) > 1e-10:
                polished_inc = polish_rational(intercept)
                if isinstance(polished_inc, Fraction) and polished_inc.denominator != 1:
                    val_str = str(polished_inc)
                else:
                    val = float(polished_inc)
                    val_str = f"{val:.10g}"
                    if abs(val - round(val)) < 1e-9:
                        val_str = str(int(round(val)))
                
                if val_str != "0":
                    if not parts:
                        parts.append(val_str)
                    else:
                         if val_str.startswith("-"):
                             parts.append(f"- {val_str.lstrip('-')}")
                         else:
                             parts.append(f"+ {val_str}")
                             
            if parts:
                return (True, " ".join(parts), None, None)
            else:
                return (True, "0", None, None)
                
    except Exception as e:
        if verbose: print(f"Linear regression failed: {e}")
        pass
        
    return (False, None, None, None)

def solve_polynomial(
    data_points: List[Tuple[Any, Any]],
    param_names: List[str],
    degree: int = 2,
    verbose: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    try:
        X_vals = []
        y_vals = []
        for p in data_points:
            x_in = p[0]
            if isinstance(x_in, (list, tuple, np.ndarray)):
                row = [float(xi) for xi in x_in]
            else:
                row = [float(x_in)]
            X_vals.append(row)
            y_vals.append(float(p[1]))
            
        X_arr = np.array(X_vals)
        y_arr = np.array(y_vals)
        
        if len(X_vals) < 3: # Need more points for poly
             return (False, None, None, "Not enough data for polynomial regression")

        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(X_arr)
        
        lr = LinearRegression(fit_intercept=False) # Bias is in X_poly
        lr.fit(X_poly, y_arr)
        
        y_pred = lr.predict(X_poly)
        mse = np.mean((y_arr - y_pred) ** 2)
        
        if mse < 1e-9:
            # Reconstruct string
            feature_names = poly.get_feature_names_out(param_names)
            parts = []
            
            for name, coef in zip(feature_names, lr.coef_):
                if abs(coef) < 1e-10: continue
                
                # Cleanup name: "x0^2" -> "x^2" if param_names used
                # sklearn output is usually "x y", "x^2", "1"
                # If we passed param_names=["x", "y"], it returns "x y"
                
                polished = polish_rational(coef)
                
                # Format Name
                # Sklearn uses space for product: "x y" -> "x*y"
                term_name = name.replace(" ", "*")
                if term_name == "1":
                    term_str = ""
                else:
                    term_str = term_name
                    
                # Combine
                val_str = ""
                if isinstance(polished, Fraction) and polished.denominator != 1:
                     val_str = str(polished)
                else:
                     val = float(polished)
                     if abs(val - round(val)) < 1e-9:
                         val_str = str(int(round(val)))
                     else:
                         val_str = f"{val:.10g}"
                
                # Logic to merge coeff and var
                full_term = ""
                if term_str == "": # Constant
                    full_term = val_str
                else:
                    if val_str == "1": full_term = term_str
                    elif val_str == "-1": full_term = f"-{term_str}"
                    else: full_term = f"{val_str}*{term_str}"
                    
                if not parts:
                    parts.append(full_term)
                else:
                    if full_term.startswith("-"):
                        parts.append(f"- {full_term[1:]}")
                    else:
                        parts.append(f"+ {full_term}")
                        
            if parts:
                return (True, " ".join(parts), None, None)
            else:
                 return (True, "0", None, None)
                 
    except Exception as e:
        if verbose: print(f"Poly regression failed: {e}")
        pass
        
    return (False, None, None, None)
