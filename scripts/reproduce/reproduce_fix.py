
import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
import sympy as sp

def verify_fix():
    print("Setting up reproduction data...")
    # Create data around singularity at x=3
    # Matches user data range approx [1, 4.5]
    X_vals = np.concatenate([
        np.linspace(1.0, 2.9, 20),
        np.linspace(3.1, 4.5, 15)
    ])
    X = X_vals.reshape(-1, 1)
    
    # Target function: sin(1/(x-3))
    # Note: explicit float division 1.0/(...)
    y = np.sin(1.0 / (X_vals - 3.0))
    
    # Inspect specific point that was problematic
    # x=3.3 -> 1/0.3 = 3.333... -> sin(3.333) ~= -0.19
    # x=2.7 -> 1/-0.3 = -3.333... -> sin(-3.333) ~= 0.19
    
    idx_33 = np.abs(X_vals - 3.3).argmin()
    idx_27 = np.abs(X_vals - 2.7).argmin()
    
    print(f"Check x={X_vals[idx_33]:.2f}, y={y[idx_33]:.4f} (Expect ~ -0.19)")
    print(f"Check x={X_vals[idx_27]:.2f}, y={y[idx_27]:.4f} (Expect ~ 0.19)")
    
    # Seed to test
    seed_str = "sin(1/(x-3))"
    
    config = GeneticConfig(
        population_size=50,
        generations=5,
        verbose=True,
        seeds=[seed_str]
    )
    
    print(f"\nInitializing Genetic Engine with seed: {seed_str}")
    reg = GeneticSymbolicRegressor(config)
    
    # Manually trigger seed evaluation logic (part of init_islands or fit)
    # We can just run fit()
    
    print("Running fit() in DIRECT space...")
    pareto = reg.fit(X, y, variable_names=["x"])
    
    if pareto:
        best = pareto.get_best()
        print(f"\nBest Solution found: {best.expression}")
        print(f"MSE: {best.mse}")
        
        if best.mse < 1e-4:
            print("SUCCESS: MSE is effectively 0.")
        else:
            print("FAILURE: MSE is too high (expected < 1e-4).")
    else:
        print("FAILURE: No solution found.")

if __name__ == "__main__":
    try:
        verify_fix()
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
