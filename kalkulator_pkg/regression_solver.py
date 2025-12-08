import warnings

import numpy as np
import sympy as sp
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler

from .function_finder_advanced import generate_candidate_features, lasso_regression
from .noise_handling.robust_regression import robust_fit


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
    except Exception:
        return None


def solve_regression_stage(
    X_data, y_data, data_points, param_names, include_transcendentals=True
):
    # --- Step 0: Filter NaNs/Infs ---
    # Robustly remove any data points with invalid values
    try:
        X_arr = np.array(X_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)
        
        # Check masks
        valid_mask_x = np.all(np.isfinite(X_arr), axis=1)
        valid_mask_y = np.isfinite(y_arr)
        valid_mask = valid_mask_x & valid_mask_y
        
        if np.sum(valid_mask) < len(y_arr):
             X_data = X_arr[valid_mask]
             y_data = y_arr[valid_mask]
             data_points = [d for i, d in enumerate(data_points) if valid_mask[i]]
    except Exception:
        pass # Fallback to original if array conversion fails (shouldn't happen)

    # Generate feature matrix
    X_matrix, feature_names = generate_candidate_features(
        X_data, param_names, include_transcendentals=include_transcendentals, y_data=y_data
    )
    
    # Pre-filtering: Detect and remove gross outliers using Robust Regression on Linear/Quadratic model
    # This comes AFTER feature generation but BEFORE fitting.
    # Actually, simplistic linear check might not work for complex functions.
    # But if we have 100 vs 6, even a robust mean would catch it.
    # Phase 0 Exact Linear Check
    # Need enough points to determine coefficients (n_vars + 1 usually).
    # But let's just try if we have > 2 points.
    if len(y_data) > 2:
         # --- PHASE 0: EXACT INTEGER LINEAR CHECK (Genius Mode) ---
         # Check if y is exactly A*x + B*y + C with integer/simple rational coefficients.
         try:
            if X_data.ndim == 2:
                X_lin = X_data
            else:
                X_lin = X_data.reshape(-1, 1)
            
            # Augment with 1 for intercept
            X_aug = np.column_stack([X_lin, np.ones(len(y_data))])
            
            # Least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, y_data, rcond=None)
            
            # Check if residuals are effectively zero
            y_pred = X_aug @ coeffs
            mse = np.mean((y_data - y_pred)**2)
            
            if mse < 1e-18: # It's linear
                # Try to snap coefficients to integers or simple rationals
                snapped_coeffs = []
                all_snapped = True
                
                for c in coeffs:
                    # Try integer
                    c_int = int(round(c))
                    if abs(c - c_int) < 1e-9:
                        snapped_coeffs.append(c_int)
                        continue
                    
                    # Try simple rational (denominator up to 12)
                    from fractions import Fraction
                    c_frac = Fraction(c).limit_denominator(12)
                    if abs(c - float(c_frac)) < 1e-9:
                        snapped_coeffs.append(c_frac)
                        continue
                    
                    all_snapped = False
                    break
                
                if all_snapped:
                    # We found an exact simple linear form!
                    terms = []
                    for i, c in enumerate(snapped_coeffs[:-1]): # Exclude intercept for now
                        if c == 0: continue
                        name = param_names[i]
                        if c == 1: term = name
                        elif c == -1: term = f"-{name}"
                        else: term = f"{c}*{name}"
                        terms.append(term)
                    
                    c_int = snapped_coeffs[-1] # Intercept
                    if c_int != 0:
                        terms.append(str(c_int))
                        
                    if not terms: terms = ["0"]
                    
                    poly_str = " + ".join(terms).replace("+ -", "- ")
                    return True, poly_str, "exact_linear", 0.0
         except Exception:
             pass
         # Use simple linear fit on X_data (first few cols of X_matrix are usually linear terms)
         # Or robust_fit on full X_matrix might be too slow if many features.
         # Let's trust robust_fit's outlier detection on the RAW X_data first?
         # But X_data might be multi-dimensional.
         try:
             # Fit a robust linear model to detect gross outliers
             # We use the raw input variables + constant
             X_check = np.array(X_data)
             if X_check.ndim == 1: X_check = X_check.reshape(-1, 1)
             
             # Add bias
             X_check_aug = np.column_stack([np.ones(len(y_data)), X_check])
             
             # Robust fit with higher min_samples (60%) to avoid 2-point fits on outliers
             _, _, info = robust_fit(
                 X_check_aug, y_data, method='ransac', min_samples=0.6
             )
             
             # If RANSAC rejected points, use the mask
             if 'inlier_mask' in info:
                 inlier_mask = info['inlier_mask']
                 n_inliers = np.sum(inlier_mask)
                 
                 # Safety: don't reject more than 50%
                 if n_inliers >= 0.5 * len(y_data) and n_inliers < len(y_data):
                     # apply mask to EVERYTHING
                     X_matrix = X_matrix[inlier_mask]
                     y_data = np.array(y_data)[inlier_mask]
                     data_points = [d for i, d in enumerate(data_points) if inlier_mask[i]]
         except Exception:
             pass

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
                penalty_factors[i] = 5.0  # Increased from 2.5 to penalize complex transcendentals

        if "sin(2*" in name or "cos(2*" in name:
            penalty_factors[i] = max(penalty_factors[i], 5.0) # Increased to prefer simple polys
        elif "sin(pi*" in name or "cos(pi*" in name:
            penalty_factors[i] = max(penalty_factors[i], 7.0) # Increased heavily

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
    exact_indices = None
    if len(data_points) <= 20:
        # OMP with Physics Boost
        X_omp = X_norm.copy()
        omp_boosts = np.ones(X_norm.shape[1])
        if include_transcendentals:
            # print(f"DEBUG FEATURES ({len(feature_names)}): {feature_names[:10]} ... {feature_names[-10:]}", flush=True)
            pass

        # --- DETECTION-TRIGGERED BOOST (Supporting Parent) ---
        # Run detection ONCE before the loop, then apply boosts based on results.
        sat_hints = {}
        curv_hints = {}
        detected_feature_idx = None  # Track the detected pattern's index
        if y_data is not None and len(y_data) >= 8 and include_transcendentals:
            try:
                from kalkulator_pkg.function_finder_advanced import detect_saturation, detect_curvature
                x_col = X_data[:, 0] if X_data.ndim > 1 else X_data
                sat_hints = detect_saturation(x_col, y_data)
                curv_hints = detect_curvature(x_col, y_data)
                
                # --- DETECTION-PRIORITY OVERRIDE ---
                # When detection strongly confirms a pattern, find that feature and
                # force-select it (trust the pattern detector over raw correlation)
                # Priority: sigmoid (double saturation = more specific) > softplus
                if sat_hints.get('sigmoid'):
                    for idx, name in enumerate(feature_names):
                        if '1/(1+exp(-' in name:
                            detected_feature_idx = idx
                            break
                elif sat_hints.get('softplus'):
                    for idx, name in enumerate(feature_names):
                        if name == 'log(1+exp(x))' or name == f'log(1+exp({param_names[0]}))':
                            detected_feature_idx = idx
                            break
            except Exception:
                pass  # Detection failed - fall back to regular search

        for i, name in enumerate(feature_names):
            # NO TRAINING WHEELS - But we boost features that match DETECTED patterns.
            # This is the "supporting parent" - not telling the child what to learn,
            # but helping prioritize what the child has already discovered.
            scale = 1.0
            
            # Boost Softplus if detected
            if sat_hints.get('softplus') and 'log(1+exp' in name:
                scale = 200.0
                
            # Boost Sigmoid if detected
            if sat_hints.get('sigmoid') and '1/(1+exp' in name:
                scale = 50.0
                
            # Boost Tanh if detected
            if sat_hints.get('tanh') and 'tanh(' in name:
                scale = 30.0
                
            # Boost Exp if curvature detected
            if curv_hints.get('exp') and 'exp(' in name and 'log' not in name:
                scale = 20.0
                
            # Boost Log if curvature detected  
            if curv_hints.get('log') and 'log(' in name and 'exp' not in name:
                scale = 20.0
            
            if scale > 1.0:
                X_omp[:, i] *= scale
            omp_boosts[i] = scale

        # --- BFSS: Brute Force Subset Search (Anti-Greedy) ---
        # For small data, check EXACT combinations of top correlated features
        # This solves finding sin(10t)+sin(11t) when OMP greedily picks sin(12t)
        coeffs = None
        exact_indices = None
        
        # --- DETECTION-PRIORITY OVERRIDE ---
        # When detection strongly confirms a pattern, force-select that feature
        # This trusts the pattern detector over raw correlation (per BotBicker debate)
        if detected_feature_idx is not None:
            # Force-select the detected feature
            exact_indices = [detected_feature_idx]
        elif len(data_points) <= 20 and X_norm.shape[1] < 500:
             # Increased limit to 500 to allow Genius Features scan
             # Pass OMP Boosts as priorities to screen features differently
             exact_indices = find_best_subset_small_data(
                 X_norm, y_centered, max_subset_size=3, top_k=60, priorities=omp_boosts
             )

        if exact_indices:
             # Manual Coefficient Calculation for selected indices
             selected_indices = exact_indices
             # Get raw coefs
             X_sel = X_norm[:, exact_indices]
             try:
                 c_vals, _, _, _ = np.linalg.lstsq(X_sel, y_centered, rcond=None)
                 coeffs = np.zeros(X_norm.shape[1])
                 for idx, val in zip(exact_indices, c_vals):
                     coeffs[idx] = val
             except:
                 exact_indices = None # Fallback

        if exact_indices is None:
            # Fallback to OMP/Lasso
            pass
        else:
             pass # Will skip the block below by checking if coeffs is not None?
             
        if exact_indices is None:
            # Reduce n_nonzero to prefer parsimony (prefer 3 terms max for sparse data, else 6)
            limit_terms = 3 if len(data_points) < 15 else 6
            n_nonzero = min(
                len(data_points) - 1, limit_terms, X_omp.shape[1]
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
        
    # --- BFSS OVERRIDE ---
    if exact_indices:
        # If Brute Force found a superior subset, override greedy results
        # Re-calculate coefficients for exact indices on unweighted data?
        # Or just use the coeffs we calculated earlier?
        # Re-calculate to be safe and consistent with flow
        X_sel = X_norm[:, exact_indices]
        try:
             c_vals, _, _, _ = np.linalg.lstsq(X_sel, y_centered, rcond=None)
             coeffs = np.zeros(X_norm.shape[1])
             for idx, val in zip(exact_indices, c_vals):
                 coeffs[idx] = val
        except:
             pass # Keep OMP/Lasso result if override fails

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
                boost = 5.0  # Reduced from 50.0
            elif "^2" in name and "exp" not in name:
                boost = 10.0  # Reduced from 100.0
            elif "^" in name and "(" not in name and not name.split("^")[-1].isdigit():
                boost = 10.0  # Boost x^x
            elif "log" in name:
                boost = 10.0  # Strong Boost for Entropy
            elif "exp" in name and ("sin" in name or "cos" in name):
                boost = 10.0  # Strong Boost for Damped
            elif "*" in name:
                boost = 5.0  # Interactions

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
        
        # Use automated robust regression instead of plain OLS
        # this handles outliers (e.g. f(3)=100) by switching to RANSAC/Huber
        refined_coeffs, intercept, _ = robust_fit(
            X_selected, y_data, method='auto', fit_intercept=True
        )

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

    refined_coeffs, intercept, _ = robust_fit(
        X_matrix[:, selected_indices], y_data, method='auto', fit_intercept=True
    )

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
        
        # Zero check: strict 1e-9 cutoff
        if abs(coeff) < 1e-9:
            continue

        sym_coeff = _symbolify_coefficient(coeff)
        # Extra check: if symbolify returned "0", skip it
        if sym_coeff == "0" or sym_coeff == "-0": 
             continue
             
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
        y_values = []
        for point in data_points:
            input_vals = point[0]
            expected = eval_to_float(point[1])
            y_values.append(expected)
            input_floats = [
                eval_to_float(v) if isinstance(v, str) else float(v) for v in input_vals
            ]
            subs_dict = {local_dict[n]: v for n, v in zip(param_names, input_floats)}
            computed = float(func_expr.subs(subs_dict))
            total_error += (computed - expected) ** 2

        mse = total_error / len(data_points)
        
        # Calculate R² for confidence
        y_mean = np.mean(y_values)
        ss_tot = sum((y - y_mean)**2 for y in y_values)
        r_squared = 1.0 - (total_error / ss_tot) if ss_tot > 1e-12 else 1.0
        
        # Confidence Awareness: Warn user if R² is low
        confidence_note = ""
        if r_squared < 0.5:
            confidence_note = " [LOW CONFIDENCE: R²={:.2f}]".format(r_squared)
        elif r_squared < 0.9:
            confidence_note = " [R²={:.2f}]".format(r_squared)
            
        return (True, func_str + confidence_note, None, mse)
    except Exception:
        return (False, func_str, None, 1e9)


def find_best_subset_small_data(X, y, max_subset_size=3, top_k=50, priorities=None):
    from itertools import combinations
    n_features = X.shape[1]
    
    # 1. Screen features by correlation with y
    correlations = []
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-9:
        return []

    for i in range(n_features):
        col = X[:, i]
        col_norm = np.linalg.norm(col)
        if col_norm < 1e-9:
            corr = 0.0
        else:
            corr = abs(np.dot(col, y) / (col_norm * y_norm))
        
        # Apply Priority Boost if provided
        if priorities is not None:
            # priorities[i] is scale (e.g. 150.0). We should not multiply lineraly maybe?
            # Actually, simply multiplying allows high-priority features to jump queue.
            # Even low correlation * 150 will beat high correlation?
            # If real correlation is 0.01 * 150 = 1.5. A pure noise is 0.1.
            # Yes, multiplying works.
            corr *= priorities[i]
            
        correlations.append((corr, i))
    
    # Sort and take top K candidates
    correlations.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for _, idx in correlations[:top_k]]
    
    best_mse = float('inf')
    best_subset = None
    
    y_var = np.var(y)
    if y_var < 1e-9: y_var = 1.0

    def _fit_mse(indices):
        X_sub = X[:, indices]
        try:
            coef, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
            pred = X_sub @ coef
            return np.mean((y - pred)**2)
        except:
            return float('inf')

    # 2. Check 1-term
    for idx in top_indices:
        mse = _fit_mse([idx])
        # Apply inverse priority penalty: lower priority features get MSE inflated
        # This favors detected features (high priority) over spurious correlations
        if priorities is not None and priorities[idx] < 10.0:
            mse *= 10.0  # Penalize non-detected features
        if mse < best_mse:
            best_mse = mse
            best_subset = [idx]
            
    # 3. Check 2-term
    if max_subset_size >= 2:
        for c in combinations(top_indices, 2):
            mse = _fit_mse(list(c))
            if mse < best_mse * 0.95: # Strict improvement required
                 best_mse = mse
                 best_subset = list(c)
                 
    # 4. Check 3-term (limit to top 25 features to keep it fast)
    if max_subset_size >= 3:
        top_25 = top_indices[:25]
        for c in combinations(top_25, 3):
             mse = _fit_mse(list(c))
             if mse < best_mse * 0.95:
                 best_mse = mse
                 best_subset = list(c)

    # 5. Acceptance Check
    # If MSE is < 1% of variance (R2 > 0.99) OR significantly better than null
    if best_mse / y_var < 0.05: # Accept decent fits
         return best_subset
    
    return None
