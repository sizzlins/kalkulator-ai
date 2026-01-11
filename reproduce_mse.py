
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def debug_manual_mse():
    print("Debugging Manual MSE for f(x)=x on ReLU data...")
    
    # f(-5)=0 ... f(0)=0 ... f(5)=5
    x_vals = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    y_vals = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    
    X = np.array(x_vals).reshape(-1, 1)
    y = np.array(y_vals)
    
    # Configure regressor to mimic user's state
    config = GeneticConfig(population_size=10, generations=1, verbose=True)
    reg = GeneticSymbolicRegressor(config)
    
    # Force 'scale invariant fitness' check logic
    y_max = np.max(np.abs(y))
    y_median = np.median(np.abs(y))
    if y_median == 0: y_median = 1e-10
    skew_ratio = y_max / y_median
    print(f"Skew Ratio: {skew_ratio} (Target > 1000 for relative fitness)")
    
    # Manually trigger relative fitness flag if needed
    if skew_ratio > 1000:
        reg._use_relative_fitness = True
        print("Using Relative Fitness")
        
    # Evaluate 'pred = x'
    pred = X.flatten()
    
    print("\n--- MSE Calculation Logic ---")
    valid = np.ones(len(y), dtype=bool) # x is valid everywhere
    
    if hasattr(reg, '_use_relative_fitness') and reg._use_relative_fitness:
        # Replicating the logic from genetic_engine.py
        denom = np.abs(y[valid])
        denom[denom < 1e-10] = 1.0 # avoid div/0
        
        diff = pred[valid] - y[valid]
        diff_rel = diff / denom
        
        print(f"Pred: {pred}")
        print(f"Y:    {y}")
        print(f"Diff: {diff}")
        print(f"Denom:{denom}")
        print(f"Rel:  {diff_rel}")
        print(f"Rel^2:{diff_rel**2}")
        
        mse = np.mean(diff_rel**2)
        print(f"Calculated MSE: {mse}")
        
    else:
        mse = np.mean((pred[valid] - y[valid])**2)
        print(f"Standard MSE: {mse}")

if __name__ == "__main__":
    debug_manual_mse()
