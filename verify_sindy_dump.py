
import numpy as np
import pandas as pd
import sys
import os
import json

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
    
    config = SINDyConfig(derivative_method="finite_difference", threshold=0.01)
    sindy = SINDy(config)
    sindy.fit(X, t, variable_names=['y'])
    
    eqs = sindy.equations
    with open('final_res.txt', 'w') as f:
        f.write(str(eqs))

if __name__ == "__main__":
    verify_sindy()
