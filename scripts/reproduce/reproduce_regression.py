
import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
import sympy as sp

def verify_regression():
    print("Setting up regression data for sin(x/(x-3))...")
    # Coverage matches user data range [-20, 20] with 257 points
    X_vals = np.linspace(-20, 20, 257)
    
    # Filter singularity at exactly 3.0
    mask = np.abs(X_vals - 3.0) > 1e-9
    X_vals = X_vals[mask]
    
    X = X_vals.reshape(-1, 1)
    
    # Target function: sin(x/(x-3))
    # y = sin(x / (x - 3))
    y = np.sin(X_vals / (X_vals - 3.0))
    
    idx_45 = np.abs(X_vals - 4.5).argmin()
    print(f"Check x={X_vals[idx_45]:.2f}, y={y[idx_45]:.4f} (Expect ~ 0.1411)")
    
    config = GeneticConfig(
        population_size=500,
        generations=10,
        verbose=True,
    )
    
    print(f"\nInitializing Genetic Engine (Forensic Analysis Enabled)...")
    reg = GeneticSymbolicRegressor(config)

    print("Running fit()...")
    # This triggers forensic_analysis -> _detect_singularity_zeros
    pareto = reg.fit(X, y, variable_names=["x"])
    
    if pareto:
        best = pareto.get_best()
        print(f"\nBest Solution found: {best.expression}")
        print(f"MSE: {best.mse}")
        
        if best.mse < 1e-4:
            print("SUCCESS: MSE is effectively 0.")
        else:
            print("FAILURE: MSE is too high.")
    else:
        print("FAILURE: No solution found.")

if __name__ == "__main__":
    try:
        verify_regression()
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
