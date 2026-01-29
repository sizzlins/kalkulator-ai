
import numpy as np
import sys
import os

# Ensure importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def test_huber_shield():
    print("[INFO] Testing Huber Loss Boosting Shield...")
    
    # 1. Generate Data with ONE massive outlier
    # Simple constant function y=0, but one outlier y=1000
    X = np.linspace(0, 10, 11).reshape(-1, 1) # 11 points
    y = np.zeros(11)
    y[5] = 1000.0 # Outlier at index 5
    
    # 2. Configure Huber
    config = GeneticConfig(
        population_size=50,
        generations=5, # Short run
        boosting_rounds=1, # Just one round to see what it fits
        loss_function="huber",
        huber_delta=1.0,
        verbose=True
    )
    
    reg = GeneticSymbolicRegressor(config)
    reg.fit(X, y)
    
    # 3. Check Prediction at Outlier
    pred = reg.predict(X)
    outlier_pred = pred[5]
    
    print(f"[RESULT] Prediction at Outlier (Target 1000): {outlier_pred}")
    
    # Analysis
    # If MSE (Wrecking Ball): Prediction will try to be mean(y) ~ 100, or huge if it finds a spike.
    # If Huber (Shield): Top gradient is capped at 1.0. 
    # So the tree can at most predict something like 1.0 * learning_rate (0.1) * booster?
    # Or purely fit the gradient 1.0.
    # If the tree finds "1", then prediction is 1.
    # If prediction < 10, we are safe.
    # If prediction > 100, we failed.
    
    if abs(outlier_pred) < 10.0:
        print("[SUCCESS] Model ignored the 1000.0 outlier! (Huber Shield Active)")
    else:
        print("[FAIL] Model chased the outlier! (Huber logic broken)")

if __name__ == "__main__":
    test_huber_shield()
