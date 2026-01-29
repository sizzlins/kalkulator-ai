
import os

target_file = r"C:\Users\LOQ\PycharmProjects\kalkulator-ai\kalkulator_pkg\symbolic_regression\genetic_engine.py"

with open(target_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
found = False

# The new body code to inject
new_body = """        # boosting loop logic moved from GeneticSymbolicRegressor
        boosted_models = []
        
        # v4.3 Audit Fix: Distinguish Physical Residual (y - F) from Target Gradient
        # Initialize Physical Residual (R_0 = y_train)
        physical_residual = y_train.copy() # Actual error: y_true - F_current(x)
        
        best_tree_final = None
        
        rounds = self.config.boosting_rounds
        if rounds < 1: rounds = 1
        
        # Helper for Loss Calculation logic
        loss_type = getattr(self.config, 'loss_function', 'mse').lower()
        huber_delta = getattr(self.config, 'huber_delta', 1.35)
        
        import time
        start_time_global = time.time()
        
        # DEBUG: Inspect data entering training
        if self.config.verbose:
            print(f"DEBUG: train_full_model data inspection:")
            print(f"  X_train shape: {X_train.shape}")
            if len(X_train) > 0:
                print(f"  X_train[0:5]: {X_train[0:5].flatten()}")
            print(f"  y_train shape: {y_train.shape}")
            if len(y_train) > 0:
                print(f"  y_train[0:5]: {y_train[0:5]}")
            if seeds:
                print(f"  Seeds received: {len(seeds)}")
        
        # Parallel Config
        # WINDOWS FIX: Force serial execution for robustness
        n_jobs = 1
        use_parallel = False
        
        for round_idx in range(rounds):
            if self.config.verbose and rounds > 1:
                print(f"--- Boosting Round {round_idx + 1}/{rounds} ---")
                
            # 1. Calculate Target for this tree (Negative Gradient)
            # pseudo_residual = - Gradient(Loss(y, F))
            # For MSE: - (F - y) = y - F = physical_residual
            # For Huber: Clip(physical_residual)
            
            if loss_type == 'huber':
                abs_r = np.abs(physical_residual)
                mask_small = abs_r <= huber_delta
                mask_large = ~mask_small
                
                target_gradient = np.zeros_like(physical_residual)
                # Small error: Gradient is residual
                target_gradient[mask_small] = physical_residual[mask_small]
                # Large error: Gradient is constant delta * sign(residual)
                target_gradient[mask_large] = huber_delta * np.sign(physical_residual[mask_large])
                
                if self.config.verbose:
                    n_outliers = np.sum(mask_large)
                    if n_outliers > 0:
                        print(f"   Huber Active: Clipped {n_outliers} outliers for training target.")
            else:
                target_gradient = physical_residual.copy()
            
            
            # 2. Train Tree on Target Gradient
            # Initialize Islands using Target
            islands = self._init_islands_internal(variable_names, X_train, target_gradient, seeds=seeds)
            
            # Run Evolution Round
            islands = self.train(
                islands, X_train, target_gradient, X_val, y_val,
                sample_weight=sample_weight,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
                start_time_global=start_time_global
            )
            
            # 3. Select Best Tree
            best_round = self.pareto_front.get_best()
             
            if not best_round:
                if self.config.verbose: print("Evolution failed to find any valid solution this round.")
                break
                
            # 4. Store Model and Update
            learning_rate = getattr(self.config, 'learning_rate', 0.1)
            boosted_models.append((learning_rate, best_round.tree))
            best_tree_final = best_round.tree
            
            # Update Physical Residual
            try:
                # Use fast evaluate
                tree_pred = best_round.tree.evaluate_fast(X_train)
                # Ensure scalar is broadcast
                if hasattr(tree_pred, 'shape') and tree_pred.shape != X_train.shape[0]:
                     pass # handled by broadcasting usually, but careful
                # Just using scalar check
                import numpy as np
                if np.isscalar(tree_pred): tree_pred = np.full(X_train.shape[0], tree_pred)
                
                # F_new = F_old + lr * T
                # R_new = y - F_new = R_old - lr * T
                physical_residual = physical_residual - (learning_rate * tree_pred)
                
                if self.config.verbose:
                    resid_mse = np.mean(physical_residual**2)
                    print(f"Round {round_idx+1} Post-Update: Physical MSE = {resid_mse:.4e}")

                # EARLY STOPPING
                if resid_mse < 1e-9:
                     if self.config.verbose: print(f"Perfect physical fit (MSE < 1e-9). Stopping boosting.")
                     break
                     
            except Exception as e:
                print(f"Boosting Update Failed: {e}")
                break
            
            # Clear Pareto Front for next round
            self.pareto_front = ParetoFront() # Reset
            
        return best_tree_final, boosted_models
"""

for line in lines:
    stripped = line.strip()
    
    # Detect start of old body
    if "# boosting loop logic moved from GeneticSymbolicRegressor" in line:
        found = True
        skip = True
        new_lines.append(new_body) # Insert all new code at once
        continue
    
    # Detect end of old body
    if skip:
        if stripped == "return best_tree_final, boosted_models":
            skip = False
            # Don't append this line because it's included in new_body (at the end)
            # Actually, I included it in new_body.
        continue # Skip old lines
    else:
        new_lines.append(line)

if found:
    print("Found and replaced boosting loop.")
    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
else:
    print("Could not find start of boosting loop.")
