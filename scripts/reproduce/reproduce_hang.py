
import numpy as np
import sys
import os
import time

# Ensure we can import the package
sys.path.append(os.getcwd())

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def reproduce():
    print("[Debug] Script Started: Reproducing hang with complex data (High Load)...")
    
    # Constructing X and y from user log - Expanded to 100 points
    # f(4.5) = 2.06887208520327
    data = [
        (4.5, 2.06887208520327),
        (4.0, 2.0),
        (1.0, 1.0),
        (-20.0, 3.02050205308471 + 1.46652089152635j),
        (-1.0, 1.69038675716359 + 1.86990796402678j),
        (0.001, -1.3876420639449 + 2.6239083670902j),
    ] * 17 # 6 * 17 = 102 points
    
    X = np.array([d[0] for d in data]).reshape(-1, 1)
    # y contains complex values, so log(y) will be complex. This triggers the path we fixed.
    y = np.array([d[1] for d in data])
    
    print(f"Data types: X={X.dtype}, y={y.dtype}, Shape={X.shape}")
    
    config = GeneticConfig(
        population_size=300,
        generations=20, 
        n_islands=2,   # Match user scenario
        timeout=15,
        verbose=True
    )
    
    reg = GeneticSymbolicRegressor(config)
    print("Calling fit (simulated Log Space)...")
    # Simulate Log Space: Transformed y = log(y)
    # This will produce complex values if y has negative/complex parts
    y_log = np.log(y.astype(complex))
    
    # Run standard fit on log-transformed data
    reg.fit(X, y_log, ["x"])
    print("Finished.")

if __name__ == "__main__":
    # Windows ProcessPoolExecutor guard
    try:
        reproduce()
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Crash: {e}")
