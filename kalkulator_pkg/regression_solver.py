import warnings

import numpy as np
import sympy as sp
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler

from .function_finder_advanced import generate_candidate_features
from .function_finder_advanced import lasso_regression
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
        if abs(val) < 1e-9:
            return None  # Suppress near-zero

        # 1. Fast Path: Integers
        try:
            rounded = round(val)
        except (TypeError, ValueError):
            return None  # Complex or other non-roundable type
        if abs(val - rounded) < 1e-9:
            if rounded == 1:
                return None  # Will be handled as implicit 1
            if rounded == -1:
                return None  # Will be handled as implicit -1
            return str(int(rounded))

        # 2. Fast Path: Simple Fractions
        # (detect_symbolic_constant covers this, but this is faster for common cases)
        for denom in [2, 3, 4, 5, 6, 8, 10, 12]:
            for num in range(-24, 25):
                if num == 0:
                    continue
                frac_val = num / denom
                if abs(val - frac_val) < 1e-9:
                    if denom == 1:
                        return str(num)
                    return f"{num}/{denom}" if num > 0 else f"({num}/{denom})"

        # 3. Robust Symbolic Detection (Pi, e, sqrt, etc.)
        from .function_finder_advanced import detect_symbolic_constant

        sym = detect_symbolic_constant(val, tolerance=1e-4)
        if sym is not None:
            s = str(sym).replace(" ", "")
            # SymPy might return "1*pi", clean it up
            s = s.replace("1*pi", "pi")
            return s

        return None
    except Exception:
        return None


def solve_regression_stage(
    X_data, y_data, data_points, param_names, include_transcendentals=True
):
    # --- Step 0: Filter NaNs/Infs ---
    # Robustly remove any data points with invalid values
    # BUT preserve original data for pole detection!
    X_original = None
    y_original = None
    try:
        X_arr = np.array(X_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)

        # Preserve original for pole detection
        X_original = X_arr.copy()
        y_original = y_arr.copy()

        # Check masks
        valid_mask_x = np.all(np.isfinite(X_arr), axis=1)
        valid_mask_y = np.isfinite(y_arr)
        valid_mask = valid_mask_x & valid_mask_y

        if np.sum(valid_mask) < len(y_arr):
            X_data = X_arr[valid_mask]
            y_data = y_arr[valid_mask]
            data_points = [d for i, d in enumerate(data_points) if valid_mask[i]]
    except Exception:
        pass  # Fallback to original if array conversion fails (shouldn't happen)

    # --- SCALE-INVARIANT NORMALIZATION ---
    # Detect data skew similar to genetic_engine.py
    # If data spans many orders of magnitude, normalize using relative values
    # This prevents bad seeds for hybrid mode
    use_relative_normalization = False
    y_scale_factor = 1.0
    
    try:
        y_arr = np.array(y_data, dtype=float)
        if len(y_arr) > 0:
            y_abs = np.abs(y_arr)
            y_median = np.median(y_abs)
            y_max = np.max(y_abs)
            
            if y_median > 0:
                skew_ratio = y_max / y_median
                
                # Same threshold as genetic_engine.py
                if skew_ratio > 1000:
                    use_relative_normalization = True
                    # Normalize by median (keeps relative proportions)
                    y_scale_factor = y_median
                    if y_scale_factor < 1e-100:
                        y_scale_factor = 1.0
                    
                    # Scale y_data to reduce range
                    y_data = y_arr / y_scale_factor
    except Exception:
        pass
    
    # Generate feature matrix
    # Pass ORIGINAL (unfiltered) data for pole detection, but filtered for regression
    X_matrix, feature_names = generate_candidate_features(
        X_data,
        param_names,
        include_transcendentals=include_transcendentals,
        y_data=y_data,  # Filtered y for frequency detection
        X_original=X_original,  # Original X for pole detection
        y_original=y_original,  # Original y for pole detection
    )

    # --- Step 0.5: Filter Invalid Features ---
    # Remove any feature columns that contain NaN or Inf values
    # This acts as a safety barrier for the Lasso solver
    try:
        valid_feature_mask = np.all(np.isfinite(X_matrix), axis=0)
        if not np.all(valid_feature_mask):
            X_matrix = X_matrix[:, valid_feature_mask]
            feature_names = [n for i, n in enumerate(feature_names) if valid_feature_mask[i]]
            
            # If all features removed, return failure
            if X_matrix.shape[1] == 0:
                 return (False, None, None, 1e9)
    except Exception:
        pass

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
            X_arr_lin = np.array(X_data, dtype=float)
            if X_arr_lin.ndim == 2:
                X_lin = X_arr_lin
            else:
                X_lin = X_arr_lin.reshape(-1, 1)

            # Augment with 1 for intercept
            X_aug = np.column_stack([X_lin, np.ones(len(y_data))])

            # Least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, y_data, rcond=None)

            # Check if residuals are effectively zero
            y_pred = X_aug @ coeffs
            mse = np.mean((y_data - y_pred) ** 2)
            
            # Calculate R² for linear fit
            y_mean_val = np.mean(y_data)
            ss_tot = np.sum((y_data - y_mean_val) ** 2)
            r_squared = 1.0 - (mse * len(y_data) / ss_tot) if ss_tot > 1e-12 else 1.0

            # Accept if EXACT fit (MSE < 1e-18) OR high R² (> 0.95) with simple coefficients
            is_good_linear_fit = mse < 1e-18 or (r_squared > 0.95 and mse < 1.0)
            
            if is_good_linear_fit:
                # Try to snap coefficients to integers or simple rationals
                snapped_coeffs = []
                all_snapped = True

                for c in coeffs:
                    # Try integer
                    try:
                        c_int = int(round(c))
                    except (TypeError, ValueError):
                         all_snapped = False
                         break
                    if abs(c - c_int) < 1e-6:  # Relaxed from 1e-9 for approximate fits
                        snapped_coeffs.append(c_int)
                        continue

                    # Try simple rational (denominator up to 12)
                    from fractions import Fraction

                    c_frac = Fraction(c).limit_denominator(12)
                    if abs(c - float(c_frac)) < 1e-6:  # Relaxed tolerance
                        snapped_coeffs.append(c_frac)
                        continue

                    all_snapped = False
                    break

                if all_snapped:
                    # We found a simple linear form!
                    terms = []
                    for i, c in enumerate(
                        snapped_coeffs[:-1]
                    ):  # Exclude intercept for now
                        if c == 0:
                            continue
                        name = param_names[i]
                        sym = _symbolify_coefficient(c)
                        if sym:
                            term = f"{sym}*{name}"
                        elif c == 1:
                            term = name
                        elif c == -1:
                            term = f"-{name}"
                        else:
                            term = f"{c}*{name}"
                        terms.append(term)

                    c_int = snapped_coeffs[-1]  # Intercept
                    if c_int != 0:
                        terms.append(str(c_int))

                    if not terms:
                        terms = ["0"]

                    poly_str = " + ".join(terms).replace("+ -", "- ")
                    # Return with R² info for approximate fits
                    if mse < 1e-18:
                        return True, poly_str, "exact_linear", 0.0
                    else:
                        return True, poly_str, f"best_linear [R²={r_squared:.4f}]", mse
                else:
                    # Coefficients don't snap to simple values, but R² is still high
                    # Return the approximate linear fit with floating-point coefficients
                    if r_squared > 0.95:
                        terms = []
                        for i, c in enumerate(coeffs[:-1]):  # Exclude intercept
                            if abs(c) < 1e-9:
                                continue
                            name = param_names[i]
                            # Format coefficient cleanly
                            if abs(c - round(c)) < 0.01:
                                c_str = str(int(round(c)))
                            else:
                                c_str = f"{c:.4g}"
                            if c_str == "1":
                                terms.append(name)
                            elif c_str == "-1":
                                terms.append(f"-{name}")
                            else:
                                terms.append(f"{c_str}*{name}")
                        
                        c_intercept = coeffs[-1]
                        if abs(c_intercept) > 0.01:
                            if abs(c_intercept - round(c_intercept)) < 0.01:
                                terms.append(str(int(round(c_intercept))))
                            else:
                                terms.append(f"{c_intercept:.4g}")
                        
                        if terms:
                            poly_str = " + ".join(terms).replace("+ -", "- ")
                            return True, poly_str, f"approx_linear [R²={r_squared:.4f}]", mse
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
            if X_check.ndim == 1:
                X_check = X_check.reshape(-1, 1)

            # Add bias
            X_check_aug = np.column_stack([np.ones(len(y_data)), X_check])

            # Robust fit with higher min_samples (60%) to avoid 2-point fits on outliers
            _, _, info = robust_fit(
                X_check_aug, y_data, method="ransac", min_samples=0.6
            )

            # If RANSAC rejected points, use the mask
            if "inlier_mask" in info:
                inlier_mask = info["inlier_mask"]
                n_inliers = np.sum(inlier_mask)

                # Safety: don't reject more than 50%
                if n_inliers >= 0.5 * len(y_data) and n_inliers < len(y_data):
                    # apply mask to EVERYTHING
                    X_matrix = X_matrix[inlier_mask]
                    y_data = np.array(y_data)[inlier_mask]
                    data_points = [
                        d for i, d in enumerate(data_points) if inlier_mask[i]
                    ]
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
            elif "LambertW" in name:
                penalty_factors[i] = 0.5  # Boost LambertW (Genius Feature)
            else:
                penalty_factors[i] = (
                    5.0  # Increased from 2.5 to penalize complex transcendentals
                )

        if "sin(2*" in name or "cos(2*" in name:
            penalty_factors[i] = max(
                penalty_factors[i], 5.0
            )  # Increased to prefer simple polys
        elif "sin(pi*" in name or "cos(pi*" in name:
            penalty_factors[i] = max(penalty_factors[i], 7.0)  # Increased heavily

        col_max = np.max(np.abs(X_matrix[:, i]))
        if col_max > 20 * y_max:
            penalty_factors[i] = max(penalty_factors[i], 5.0)

        # Physics Bias Scale
        # 1. Interactions are complex -> Penalize (but with exceptions for physics patterns)
        if "*" in name:
            # SPECIAL CASE: exp()*exp() interactions represent exp(x+y) thermal/diffusion laws
            # These should be BOOSTED, not penalized
            import re
            exp_exp_pattern = re.match(r"^exp\([^)]+\)\*exp\([^)]+\)$", name)
            exp_trig_pattern = re.match(r"^exp\([^)]+\)\*(sin|cos)\([^)]+\)$", name) or \
                               re.match(r"^(sin|cos)\([^)]+\)\*exp\([^)]+\)$", name)
            
            if exp_exp_pattern:
                # exp(x)*exp(y) -> STRONG BOOST (thermal/diffusion coupling)
                penalty_factors[i] *= 0.1  # 10x boost
            elif exp_trig_pattern:
                # exp(x)*sin(y) -> MODERATE BOOST (damped oscillation)
                penalty_factors[i] *= 0.3
            else:
                # Generic interactions are complex -> Penalize
                penalty_factors[i] *= 2.0

        # 2. Rationals are physical (Inverse laws) -> Boost
        if "/" in name:
            penalty_factors[i] *= 0.5  # Moderate Boost

        # 3. Squares (x^2)
        if "^2" in name:
            if "*" not in name and "/" not in name:
                # Pure square (A^2) -> Strong Boost
                penalty_factors[i] *= 0.2
            elif "/" in name:
                # Inverse square (1/r^2) -> Boost
                penalty_factors[i] *= 0.5
            # Else: Interaction square (B*C^2) -> Leave as is (retains * penalty)

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

        # Apply the same penalty/boost logic as Lasso
        # penalty < 1.0 means Boost > 1.0
        omp_boosts = 1.0 / penalty_factors

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
                from kalkulator_pkg.function_finder_advanced import detect_curvature
                from kalkulator_pkg.function_finder_advanced import detect_saturation

                x_col = X_data[:, 0] if X_data.ndim > 1 else X_data
                sat_hints = detect_saturation(x_col, y_data)
                curv_hints = detect_curvature(x_col, y_data)

                # --- DETECTION-PRIORITY OVERRIDE (with quality check) ---
                # When detection strongly confirms a pattern, find that feature and
                # force-select it IF it actually fits well (R² > 0.8)
                # Priority: sigmoid (double saturation = more specific) > softplus
                candidate_idx = None
                if sat_hints.get("sigmoid"):
                    for idx, name in enumerate(feature_names):
                        if "1/(1+exp(-" in name:
                            candidate_idx = idx
                            break
                elif sat_hints.get("softplus"):
                    for idx, name in enumerate(feature_names):
                        if (
                            name == "log(1+exp(x))"
                            or name == f"log(1+exp({param_names[0]}))"
                        ):
                            candidate_idx = idx
                            break

                # Quality check: only override if detected feature actually fits well
                if candidate_idx is not None:
                    try:
                        feature_col = X_norm[:, candidate_idx]
                        # Simple R² check: does this feature explain variance?
                        corr = np.corrcoef(feature_col, y_centered)[0, 1]
                        if abs(corr) > 0.9:  # High correlation = good fit
                            detected_feature_idx = candidate_idx
                        # else: detection fired but feature doesn't fit well, skip override
                    except Exception:
                        pass
            except Exception:
                pass  # Detection failed - fall back to regular search

        for i, name in enumerate(feature_names):
            # NO TRAINING WHEELS - But we boost features that match DETECTED patterns.
            # This is the "supporting parent" - not telling the child what to learn,
            # but helping prioritize what the child has already discovered.
            scale = 1.0

            # Boost Softplus if detected
            if sat_hints.get("softplus") and "log(1+exp" in name:
                scale = 200.0

            # Boost Sigmoid if detected
            if sat_hints.get("sigmoid") and "1/(1+exp" in name:
                scale = 50.0

            # Boost Tanh if detected
            if sat_hints.get("tanh") and "tanh(" in name:
                scale = 30.0

            # Boost Exp if curvature detected
            if curv_hints.get("exp") and "exp(" in name and "log" not in name:
                scale = 20.0

            # Boost Log if curvature detected
            if curv_hints.get("log") and "log(" in name and "exp" not in name:
                scale = 20.0

            if scale > 1.0:
                omp_boosts[i] *= scale

        # Apply the final boosts to X_omp columns
        # OMP selects based on dot product, so scaling up the column increases its selection probability
        for i in range(X_omp.shape[1]):
            if omp_boosts[i] != 1.0:
                X_omp[:, i] *= omp_boosts[i]

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
                X_norm,
                y_centered,
                feature_names=feature_names,
                top_k=60,
                priorities=omp_boosts,
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
            except Exception:
                exact_indices = None  # Fallback

        if exact_indices is None:
            # Fallback to OMP/Lasso
            pass
        else:
            pass  # Will skip the block below by checking if coeffs is not None?

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
        except Exception:
            pass  # Keep OMP/Lasso result if override fails

    # If OMP fails or returns None (e.g., variance is 0 => Constant function),
    # we must handle it gracefully.
    if coeffs is None:
        # Fallback: Assume no features selected (or handle constant mean logic elsewhere)
        # But if we are here, we likely have a constant target.
        # Let's return empty selection which usually defaults to intercept IF intercept is implicit,
        # but here we might just want to return "failed" for this stage or "empty model".
        # Actually, if coeffs is None, usually it means we couldn't find a better fit than mean.
        # Note: y_centered is available here.
        {
            "name": "constant_fallback",
            "mse": np.mean(
                y_centered**2
            ),  # Mean of squared centered values is variance
            "r2": 0.0,
            "complexity": 1.0,
            "coefficients": np.array([0.0]),  # Coefficient for centered data is 0
            "feature_names": ["1"],
            "feature_indices": [],
            "sympy_obj": sp.Float(0.0),  # Predict 0 offset from mean
            "is_constant": True,
        }
        # solve_regression_stage expected return:
        # return success, func_str, confidence_note, mse
        # For constant function 3: y_mean is 3. Fallback returns 0 (offset).
        # We need to construct the final string here or return failure?
        # Actually, if we return failure (False, ...), function_manager tries next stage.
        # But if constant is the answer, we should return Success?
        # Wait, if `coeffs` is None, it means no features helped. So the answer IS the mean.
        # So: y = y_mean.
        # Let's return Success.
        final_func_str = f"{y_mean:.10g}"
        return (
            True,
            final_func_str,
            "Constant function (Variance=0 or no features)",
            0.0,
        )

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

    for _ in range(3):
        X_selected = X_matrix[:, selected_indices]

        # Use automated robust regression instead of plain OLS
        # this handles outliers (e.g. f(3)=100) by switching to RANSAC/Huber
        refined_coeffs, intercept, _ = robust_fit(
            X_selected, y_data, method="auto", fit_intercept=True
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
        X_matrix[:, selected_indices], y_data, method="auto", fit_intercept=True
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
                "LambertW": sp.LambertW,
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
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        r_squared = 1.0 - (total_error / ss_tot) if ss_tot > 1e-12 else 1.0

        # --- RESIDUAL-BASED PATTERN DETECTION ---
        # Only analyze residuals for REALLY bad fits (R² < 0.7)
        # This avoids cluttering output for acceptable fits
        residual_hint = ""
        if r_squared < 0.7 and len(y_values) >= 8:
            try:
                from kalkulator_pkg.function_finder_advanced import detect_frequency
                from kalkulator_pkg.function_finder_advanced import detect_saturation

                # Compute residuals
                residuals = []
                x_vals = []
                for point in data_points:
                    input_vals = point[0]
                    expected = eval_to_float(point[1])
                    input_floats = [
                        eval_to_float(v) if isinstance(v, str) else float(v)
                        for v in input_vals
                    ]
                    subs_dict = {
                        local_dict[n]: v for n, v in zip(param_names, input_floats)
                    }
                    computed = float(func_expr.subs(subs_dict))
                    residuals.append(expected - computed)
                    x_vals.append(input_floats[0] if input_floats else 0)

                residuals = np.array(residuals)
                x_vals = np.array(x_vals)

                # Quick checks for obvious missed patterns
                # PRIORITY: Specific hints (pole+trig) before generic hints (trig terms)

                # --- SMART HINT: Pole + Oscillation → Trig Composite ---
                # When we detect a pole AND the data oscillates, suggest sin(c/(x-a))
                try:
                    # Check for poles in ORIGINAL (unfiltered) data
                    # y_original and X_original contain nan/inf values that were filtered out
                    pole_x = None
                    if y_original is not None and X_original is not None:
                        for i, yv in enumerate(y_original):
                            if not np.isfinite(yv):
                                # Get x value from original X
                                if X_original.ndim == 1:
                                    pole_x = float(X_original[i])
                                elif X_original.ndim == 2:
                                    pole_x = float(X_original[i, 0])
                                break

                    # Check for oscillation (sign changes in filtered y values)
                    finite_vals = [v for v in y_values if np.isfinite(v)]
                    if len(finite_vals) > 3:
                        sign_changes = sum(
                            1
                            for i in range(len(finite_vals) - 1)
                            if finite_vals[i] * finite_vals[i + 1] < 0
                        )
                        has_oscillation = sign_changes >= 3

                        if pole_x is not None and has_oscillation:
                            residual_hint = f" [Hint: Try sin(c/(x-{pole_x})) or cos(c/(x-{pole_x}))]"
                except Exception:
                    pass

                # Generic frequency hint (only if no specific hint yet)
                if not residual_hint:
                    try:
                        freq_hints = detect_frequency(x_vals, residuals)
                        if freq_hints:
                            residual_hint = " [Hint: try adding trig terms]"
                    except Exception:
                        pass

                # Saturation hint (only if no hints yet)
                if not residual_hint:
                    try:
                        sat_hints = detect_saturation(x_vals, residuals)
                        if sat_hints.get("softplus") or sat_hints.get("sigmoid"):
                            residual_hint = " [Hint: try sigmoid/softplus]"
                    except Exception:
                        pass
            except Exception:
                pass

        # Confidence Awareness: Warn user if R² is low
        confidence_note = ""
        if r_squared < 0.5:
            confidence_note = f" [LOW CONFIDENCE: R²={r_squared:.2f}]" + residual_hint
        elif r_squared < 0.9:
            confidence_note = f" [R²={r_squared:.2f}]"

        # De-scale if we used relative normalization
        if use_relative_normalization and y_scale_factor != 1.0:
            # Multiply the entire expression by the scale factor
            try:
                func_str = f"{y_scale_factor}*({func_str})"
                # Simplify if possible
                local_dict_simple = {name: sp.Symbol(name) for name in param_names}
                local_dict_simple.update({
                    "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                    "log": sp.log, "sinh": sp.sinh, "cosh": sp.cosh,
                    "LambertW": sp.LambertW,
                })
                func_expr_scaled = sp.sympify(func_str, locals=local_dict_simple)
                func_expr_scaled = sp.simplify(func_expr_scaled)
                func_str = str(func_expr_scaled)
            except Exception:
                # If simplification fails, keep the scaled version
                pass

        return (True, func_str, confidence_note, mse)
    except Exception:
        return (False, func_str, None, 1e9)


def find_best_subset_small_data(X, y, max_subset_size=3, top_k=50, priorities=None):

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


def find_best_subset_small_data(
    X, y, feature_names=None, max_subset_size=3, top_k=50, priorities=None
):
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
            corr *= priorities[i]

        correlations.append((corr, i))

    # Sort and take top K candidates
    correlations.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for _, idx in correlations[:top_k]]
    # print(f"DEBUG: Top indices: {top_indices[:10]}", flush=True)

    best_mse = float("inf")
    best_subset = None

    y_var = np.var(y)
    if y_var < 1e-9:
        y_var = 1.0

    def _fit_mse(indices):
        X_sub = X[:, indices]
        try:
            coef, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
            pred = X_sub @ coef
            mse_val = np.mean((y - pred) ** 2)
            if mse_val < 1e-15:
                mse_val = 1e-15

            # --- Complexity Penalty (Occam's Razor) ---
            # Breaking Ties: If MSEs are very close ("perfect fits"), prefer simpler features.
            # We inflate MSE of complex features so they lose against simpler ones.
            penalty = 1.0
            if feature_names:
                for idx in indices:
                    name = feature_names[idx]
                    # Rationals (1/...) -> +10% penalty
                    if "/" in name:
                        penalty *= 1.1
                    # Transcendentals (exp, sin, log, tanh) -> +20% penalty
                    if any(
                        t in name for t in ["exp(", "log(", "sin(", "cos(", "tanh("]
                    ):
                        penalty *= 1.2
                    # Complex Interactions (x*y)...
                    if "*" in name and "^2" not in name:  # Interactions
                        penalty *= 1.05
                    # High power (x^10) -> +50% penalty
                    if "^" in name:
                        try:
                            # e.g. x^10
                            base, power = name.split("^", 1)
                            # Handle parenthesis like (x-c)^2
                            power_val = float(
                                "".join(c for c in power if c.isdigit() or c == ".")
                            )
                            if power_val >= 4:
                                penalty *= 1.5
                        except Exception:
                            pass

            return mse_val * penalty
        except Exception:
            return float("inf")

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
            subset = list(c)
            mse = _fit_mse(subset)
            if (
                mse < best_mse * 0.99
            ):  # Strictness (changed 0.95 -> 0.99 to find similar)
                best_mse = mse
                best_subset = subset

    # 4. Check 3-term (limit to top 25 features to keep it fast)
    if max_subset_size >= 3:
        top_25 = top_indices[:25]
        for c in combinations(top_25, 3):
            subset = list(c)
            mse = _fit_mse(subset)
            if mse < best_mse * 0.99:
                best_mse = mse
                best_subset = subset

    # 5. Acceptance Check
    if best_mse / y_var < 0.05:
        return best_subset

    return None
