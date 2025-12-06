import warnings

import numpy as np
import sympy as sp
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler

from .function_finder_advanced import generate_candidate_features, lasso_regression


def eval_to_float(val):
    if isinstance(val, (float, int)):
        return float(val)
    try:
        return float(val)
    except Exception:
        pass
    try:
        val_str = str(val).replace(" ", "")
        # Handle simple fractions
        if "/" in val_str and "(" not in val_str:
            n, d = val_str.split("/")
            return float(n) / float(d)
        import sympy

        return float(sympy.sympify(val).evalf())
    except Exception:
        return 0.0


def _symbolify_coefficient(val):
    try:
        if abs(val) < 1e-6:
            return None  # Suppress near-zero coefficients (fixes 0*pi)

        # Check for near-integer values first (snaps -4.999 to -5)
        rounded = round(val)
        if abs(val - rounded) < 0.001 and abs(rounded) > 0.5:
            if rounded == 1:
                return None  # Will be handled specially
            if rounded == -1:
                return None  # Will be handled specially
            return str(int(rounded))

        # Check for simple fractions (like 1/3)
        for denom in [2, 3, 4, 5, 6, 8, 10]:
            for num in range(-20, 21):
                if num == 0:
                    continue
                frac_val = num / denom
                if abs(val - frac_val) < 0.001:
                    if denom == 1:
                        return str(num)
                    return f"{num}/{denom}" if num > 0 else f"({num}/{denom})"

        # Check for common Pi fractions explicitly (4/3*pi, 1/3*pi, 1/2*pi, etc.)
        pi_val = float(sp.pi.evalf())
        for denom in [1, 2, 3, 4, 6]:
            for num in range(-15, 16):
                if num == 0:
                    continue
                expected = (num / denom) * pi_val
                if abs(val - expected) < 0.001:  # Loosened tolerance
                    if denom == 1:
                        if num == 1:
                            return "pi"
                        if num == -1:
                            return "-pi"
                        return f"{num}*pi"
                    else:
                        return f"{num}/{denom}*pi" if num > 0 else f"({num}/{denom})*pi"

        # Fallback: General Pi multiples using sympy
        ratio_pi = val / pi_val
        simplified_pi = sp.nsimplify(ratio_pi, tolerance=1e-4, rational=True)
        if simplified_pi != ratio_pi:
            den = sp.denom(simplified_pi)
            num = sp.numer(simplified_pi)
            if abs(den) < 100 and abs(num) < 100 and num != 0:
                return f"{simplified_pi}*pi".replace("1*pi", "pi")
        return None
    except:
        return None


def solve_regression_stage(
    X_data, y_data, data_points, param_names, include_transcendentals=True
):
    # Generate feature matrix
    X_matrix, feature_names = generate_candidate_features(
        X_data, param_names, include_transcendentals=include_transcendentals
    )

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_matrix)

    y_mean = np.mean(y_data)
    y_centered = np.array(y_data) - y_mean

    y_max = np.max(np.abs(y_data))
    if y_max < 1e-6:
        y_max = 1.0

    penalty_factors = np.ones(X_norm.shape[1])
    for i in range(X_norm.shape[1]):
        name = feature_names[i]
        if "exp(" in name:
            if "exp(-" in name and "^2)" in name:
                penalty_factors[i] = 1.0
            elif "exp(-" in name:
                penalty_factors[i] = 1.5
            else:
                penalty_factors[i] = 2.5

        if "sin(2*" in name or "cos(2*" in name:
            penalty_factors[i] = max(penalty_factors[i], 2.0)
        elif "sin(pi*" in name or "cos(pi*" in name:
            penalty_factors[i] = max(penalty_factors[i], 3.0)

        col_max = np.max(np.abs(X_matrix[:, i]))
        if col_max > 20 * y_max:
            penalty_factors[i] = max(penalty_factors[i], 5.0)

        # Physics Bias: Reduce penalty for Rationals and Squares
        if "/" in name:
            penalty_factors[i] *= 0.1
        elif "^2" in name:
            penalty_factors[i] *= 0.5

    X_weighted = X_norm / penalty_factors
    y_std = np.std(y_data)
    if y_std < 1e-6:
        y_std = 1.0
    adaptive_alpha = 0.01 * y_std
    adaptive_threshold = 1e-3 * y_std

    coeffs = None
    if len(data_points) < 20:
        # OMP with Physics Boost
        X_omp = X_norm.copy()
        omp_boosts = np.ones(X_norm.shape[1])
        for i, name in enumerate(feature_names):
            scale = 1.0

            # --- TIER 0: TRIPLE PRODUCTS (Highest Priority for m*g*h) ---
            # Count asterisks to detect triple products like "m*g*h"
            asterisk_count = name.count("*")
            has_power = "^" in name
            has_transcendental = (
                "sin" in name
                or "cos" in name
                or "exp" in name
                or "log" in name
                or "sinh" in name
                or "cosh" in name
            )
            has_ratio = "/" in name

            if (
                asterisk_count == 2
                and not has_power
                and not has_transcendental
                and not has_ratio
            ):
                # Pure triple product like m*g*h
                scale = 300.0
            elif (
                asterisk_count == 1
                and not has_power
                and not has_transcendental
                and not has_ratio
            ):
                # Pure double product like I*R, m*a
                scale = 200.0

            # --- TIER 1: RATIONAL PRODUCTS (A*B/C, Coulomb) ---
            elif (
                "/" in name and name.count("*") >= 2 and "^4" in name
            ):  # Triple Product Inverse Quartic (mu*L*Q/r^4)
                scale = 100.0
            elif (
                "/" in name and name.count("*") >= 2
            ):  # Triple Product Ratio (rho*u*L/mu)
                scale = 100.0
            elif "sin(" in name and "/" in name:  # Sinc function (sin(x)/x)
                scale = 60.0
            elif "^4" in name and "/" in name:  # Inverse Quartic (1/r^4)
                scale = 100.0
            elif (
                "/" in name and "*" in name
            ):  # Generalized Ratio (A*B/C or A*B/C^2) - CHECK FIRST before generic ^2
                if "^2" in name:  # Inverse Square Ratio (A*B/C^2) - COULOMB
                    scale = 150.0
                else:
                    scale = 120.0
            elif "/" in name:
                scale = 10.0

            # --- TIER 2: PURE TRIG (Higher than damped oscillation) ---
            elif name in [f"sin({v})" for v in param_names] or name in [
                f"cos({v})" for v in param_names
            ]:
                # Pure sin(x) or cos(x) - prioritize over complex damped forms
                scale = 5.0 # Reduced from 150.0

            # --- TIER 3: POLYNOMIAL INTERACTIONS ---
            elif (
                "^2" in name and "*" in name and "exp" not in name
            ):  # Poly Interactions (r^2*h) - conservative
                scale = 15.0  # Will be overridden by Structural Boosting below
            elif "^2" in name and "exp" not in name:
                scale = 10.0
            elif (
                "^" in name
                and "(" not in name
                and "^2" not in name
                and "^3" not in name
                and "^4" not in name
            ):  # Boost x^x only
                scale = 10.0
            elif "log" in name:  # Strong boost for Entropy (x*log(x))
                scale = 5.0 # Reduced from 120.0
            elif "exp" in name and (
                "sin" in name or "cos" in name
            ):  # Damped Oscillation
                scale = 5.0  # Reduced from 80.0

            # --- TIER 4: OCCAM'S RAZOR - LINEAR TERMS FOR SPARSE DATA ---
            # For very sparse data (<=5 points), strongly prefer linear terms
            n_points = len(data_points)
            if n_points <= 5 and not has_transcendental and not has_ratio:
                # Check if it's a linear term (just a variable name like "x", "t", "m")
                if name in param_names:
                    scale = max(
                        scale, 1.0
                    )  # Disabled (was 50.0) - Pre-checks handle linear cheats now
                # Also boost constant term slightly
                elif name == "1":
                    scale = max(scale, 1.0)

            # --- STRUCTURAL BOOSTING (Variable Agnostic) ---
            # Overrides for polynomial terms to prioritize simple physics shapes.
            # Applies to non-transcendental, non-rational terms.
            is_transcendental = (
                "sin" in name or "cos" in name or "exp" in name or "log" in name
            )
            if not is_transcendental and "/" not in name:
                # Estimate Degree
                # Count variables (splitting by *)
                vars_in_term = name.split("*")
                degree = 0
                for v in vars_in_term:
                    if "^" in v:
                        if "^2" in v:
                            degree += 2
                        elif "^3" in v:
                            degree += 3
                        else:
                            degree += 4  # High degree
                # Apply Boosts - Squares > Interactions (prevents Spacetime leakage)
                if degree == 2:
                    if "*" not in name:  # Pure Degree 2 (x^2) - SPACETIME PRIORITY
                        scale = max(scale, 120.0)
                    else:  # Interaction Degree 2 (x*y)
                        scale = max(scale, 100.0)
                elif degree == 3:
                    # Tier 3: Penalty Box
                    scale = max(scale, 90.0)
                # Degree 1 (Linear) left at default (or previous scale)

            if scale > 1.0:
                # if scale >= 30.0: print(f"DEBUG BOOST: name={name}, scale={scale}", flush=True)
                X_omp[:, i] *= scale
                omp_boosts[i] = scale

        # Reduce n_nonzero to prefer parsimony (prefer 6 terms max unless data requires more)
        n_nonzero = min(
            len(data_points) - 1, 6, X_omp.shape[1]
        )  # Can't request more atoms than features
        if n_nonzero < 1:
            n_nonzero = 1

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            omp.fit(X_omp, y_centered)

        coeffs = omp.coef_ * omp_boosts
    else:
        coeffs = lasso_regression(
            X_weighted.tolist(),
            y_centered.tolist(),
            lambda_reg=adaptive_alpha,
            max_iterations=50000,
        )
        coeffs = [c * p for c, p in zip(coeffs, penalty_factors)]

    selected_indices = [i for i, c in enumerate(coeffs) if abs(c) > adaptive_threshold]
    max_features = min(len(data_points) - 1, 12)
    if max_features < 1:
        max_features = 1

    if len(selected_indices) > max_features:

        def get_sort_metric(idx):
            val = abs(coeffs[idx])
            name = feature_names[idx]
            boost = 1.0
            if "/" in name:
                boost = 50.0  # Reduced boost for Rationals to avoid distracting from valid Polys
            elif "^2" in name and "exp" not in name:
                boost = 100.0  # Only boost clean squares, not Gaussians
            elif "^" in name and "(" not in name and not name.split("^")[-1].isdigit():
                boost = 100.0  # Boost x^x, ignore x^3
            elif "log" in name:
                boost = 100.0  # Strong Boost for Entropy
            elif "exp" in name and ("sin" in name or "cos" in name):
                boost = 100.0  # Strong Boost for Damped
            elif "*" in name:
                boost = 10.0  # Interactions

            # Penalize complex mixed powers if not caught above?
            if "exp(-" in name and "^2" in name:  # Gaussian
                boost = (
                    1.0  # Reset to baseline (don't penalize too much, but don't boost)
                )
            return val * boost

        selected_indices.sort(key=get_sort_metric, reverse=True)
        selected_indices = selected_indices[:max_features]

    if not selected_indices:
        return (False, None, None, 1e9)

    # Refit OLS
    refined_coeffs = None
    intercept = 0.0
    ols = None

    for _ in range(3):
        X_selected = X_matrix[:, selected_indices]
        ols = LinearRegression(fit_intercept=True)
        ols.fit(X_selected, y_data)
        refined_coeffs = ols.coef_
        intercept = ols.intercept_

        max_coeff_abs = (
            max(abs(c) for c in refined_coeffs) if len(refined_coeffs) > 0 else 0
        )
        if max_coeff_abs < 1e-12:
            break

        new_indices = []
        param_changed = False
        for i, c in enumerate(refined_coeffs):
            if abs(c) > 1e-2 * max_coeff_abs and abs(c) > 1e-8:
                new_indices.append(selected_indices[i])
            else:
                param_changed = True

        if not param_changed:
            break
        selected_indices = new_indices
        if not selected_indices:
            break

    if not selected_indices:
        return (False, None, None, 1e9)

    refined_coeffs = ols.coef_
    intercept = ols.intercept_

    # Zero Snap: If intercept is tiny relative to y_mean, snap to zero
    y_mean = np.mean(y_data)
    if abs(intercept) < 0.01 * abs(y_mean) or abs(intercept) < 1.0:
        intercept = 0.0

    equation_parts = []
    sym_intercept = _symbolify_coefficient(intercept)
    if sym_intercept:
        equation_parts.append(sym_intercept)
    elif abs(intercept) > 1e-4:
        equation_parts.append(
            f"{intercept:.10g}"
        )  # Increased threshold to suppress noise

    for i, idx in enumerate(selected_indices):
        coeff = refined_coeffs[i]
        name = feature_names[idx]
        sym_coeff = _symbolify_coefficient(coeff)
        if sym_coeff:
            equation_parts.append(f"{sym_coeff}*{name}")
        elif abs(coeff - 1.0) < 1e-9:
            equation_parts.append(name)
        elif abs(coeff + 1.0) < 1e-9:
            equation_parts.append(f"-{name}")
        else:
            equation_parts.append(f"{coeff:.10g}*{name}")

    if not equation_parts:
        func_str = "0"
    else:
        func_str = " + ".join(equation_parts).replace("+ -", "- ")

    # Verify MSE
    try:
        local_dict = {name: sp.Symbol(name) for name in param_names}
        local_dict.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "exp": sp.exp,
                "log": sp.log,
                "sinh": sp.sinh,
                "cosh": sp.cosh,
            }
        )
        func_expr = sp.sympify(func_str, locals=local_dict)

        total_error = 0
        for point in data_points:
            input_vals = point[0]
            expected = eval_to_float(point[1])
            input_floats = [
                eval_to_float(v) if isinstance(v, str) else float(v) for v in input_vals
            ]
            subs_dict = {local_dict[n]: v for n, v in zip(param_names, input_floats)}
            computed = float(func_expr.subs(subs_dict))
            total_error += (computed - expected) ** 2

        mse = total_error / len(data_points)
        return (True, func_str, refined_coeffs, mse)
    except Exception:
        return (False, func_str, refined_coeffs, 1e9)
