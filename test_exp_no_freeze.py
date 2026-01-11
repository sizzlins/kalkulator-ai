"""Test exp(3x) after fixing safe_pow to confirm no freeze."""
import sys
sys.path.insert(0, r'C:\Users\LOQ\PycharmProjects\kalkulator-ai')

import numpy as np
import time
from kalkulator_pkg.symbolic_regression import GeneticConfig, GeneticSymbolicRegressor

# User's exp(3x) data
X = np.array([[2], [2.5], [3], [3.5], [4], [4.25], [4.5], [4.8], [5], 
              [5.1], [5.3], [5.5], [5.6], [5.8], [5.9], [6], [6.5], [7]])
y = np.array([403.428793492735, 1808.04241445606, 8103.08392757538, 36315.5026742466,
              162754.791419004, 344551.896137824, 729416.369847701, 1794074.77260621,
              3269017.37247211, 4412711.89235044, 8040485.29975851, 14650719.4289535,
              19776402.6584977, 36034955.0881416, 48642101.5063338, 65659969.1373305,
              294267566.041509, 1318815734.48321])

print("Testing exp(3x) with FIXED safe_pow (no freeze)...")
print(f"Data: {len(X)} points")
print()

# Use the EXACT config from user's 'all' command
config = GeneticConfig(
    population_size=300,  # --boost 3
    generations=90,       # --boost 3
    verbose=True,
    parsimony_coefficient=0.001,
    n_islands=2,
    timeout=45,           # --boost 3
)

print("Starting evolution (should NOT freeze)...")
start_time = time.time()

regressor = GeneticSymbolicRegressor(config)

try:
    pareto = regressor.fit(X, y, ['x'])
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed in {total_time:.2f}s (NO FREEZE!)")
    
    best = pareto.get_best()
    if best:
        print(f"Result: {best.expression}")
        print(f"MSE: {best.mse:.6e}")
        
        # Check if it found exp(3x)
        if 'exp' in best.expression and '3' in best.expression:
            print("\n🎉 Found exp(3x) correctly!")
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n❌ Interrupted after {elapsed:.2f}s - freeze still exists")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
