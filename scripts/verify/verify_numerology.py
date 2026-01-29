
import numpy as np
import sys
import os

# Ensure importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def test_no_snapping():
    print("[INFO] Testing for Integer Snapping Bias...")
    
    # 1. Generate Data: y = 3.042 * x
    # The audit says: "Your Code: Oh, 3.042 is within 5% of 3. Let's just pretend it is 3."
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 3.042 * X.flatten()
    
    config = GeneticConfig(
        population_size=100,
        generations=20,
        verbose=False,
        seeds=["3*x"] # Bait it with the integer seed
    )
    
    reg = GeneticSymbolicRegressor(config)
    reg.fit(X, y)
    
    best_expr = reg.get_expression()
    print(f"[RESULT] Found: {best_expr}")
    
    # Check if it snapped to exactly 3 or 3.0
    # A float like 3.04... is good. "3*x" or "3.0*x" is bad.
    
    if "3.04" in best_expr:
        print("[SUCCESS] Engine correctly identified 3.042")
    elif best_expr.strip() == "3*x0":
        print("[FAIL] Engine snapped 3.042 to 3! Numerology detected.")
    else:
        print(f"[INFO] Found something else: {best_expr}")
        # Evaluate MSE
        y_pred = reg.predict(X)
        mse = np.mean((y - y_pred)**2)
        print(f"[INFO] MSE: {mse}")
        if mse < 1e-4:
            print("[SUCCESS] MSE is low, model is accurate.")
        else:
            print("[FAIL] Model inaccurate.")

def test_constant_reconstruction():
    print("\n[INFO] Testing Safe Constant Reconstruction...")
    # 2. Generate Data: y = 3.14159... * x (pi * x)
    # We want to see if it finds 'pi' or '3.14...'
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = np.pi * X.flatten()
    
    config = GeneticConfig(
        population_size=100,
        generations=20,
        verbose=False
    )
    reg = GeneticSymbolicRegressor(config)
    reg.fit(X, y)
    
    best_expr = reg.get_expression()
    print(f"[RESULT] Found: {best_expr}")
    
    if "pi" in best_expr:
         print("[SUCCESS] Engine safely reconstructed 'pi'.")
    elif "3.14" in best_expr:
         print("[SUCCESS] Engine found float approximation (acceptable science).")
    elif "22/7" in best_expr:
         print("[FAIL] Engine stuck in Rational Trap (22/7)!")
    else:
         print(f"[INFO] Result: {best_expr}")

if __name__ == "__main__":
    test_no_snapping()
    test_constant_reconstruction()
