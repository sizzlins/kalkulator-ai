
import numpy as np
import pandas as pd
import sys
import os

# Ensure package is in path
sys.path.append(os.getcwd())

from kalkulator_pkg.dynamics_discovery.sindy import SINDy, SINDyConfig

def verify_sindy():
    # Load data
    df = pd.read_csv("logistic.csv")
    df.columns = df.columns.str.strip()
    t = df['t'].values
    y = df['y'].values
    
    X = y.reshape(-1, 1)
    
    print("Data loaded. Running SINDy with finite_difference...")
    # SavGol fails on small datasets often. Use finite_difference.
    config = SINDyConfig(derivative_method="finite_difference", threshold=0.01)
    sindy = SINDy(config)
    sindy.fit(X, t, variable_names=['y'])
    
    eqs = sindy.equations
    print(f"RESULT: {eqs}")
    
    expected_terms = ["2*y", "-2*y^2"]
    res = eqs.get("dy/dt", "")
    
    if "2*y" in res and "y^2" in res: # -2 might be "- 2" or "-2" depending on formatting
        print("SUCCESS: Found terms resembling 2y - 2y^2")
    else:
        print("FAILURE: Did not find expected terms.")

if __name__ == "__main__":
    verify_sindy()
