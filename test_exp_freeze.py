"""Test exp(3x) with generation timing to identify freeze."""
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

print("Testing exp(3x) evolution with timing...")
print(f"Data: {len(X)} points, range {y.min():.2e} to {y.max():.2e}")
print()

# Modest config to test freezing
config = GeneticConfig(
    population_size=100,  # Smaller to test faster
    generations=10,
    verbose=True,
    parsimony_coefficient=0.001,
    n_islands=2,
)

print("Starting evolution...")
start_time = time.time()
last_gen_time = start_time

regressor = GeneticSymbolicRegressor(config)

# Patch to add timing
original_evolve = regressor._evolve_population
def timed_evolve(population, X, y, generation, sample_weight=None):
    global last_gen_time
    gen_start = time.time()
    elapsed_since_last = gen_start - last_gen_time
    print(f"\n>>> Gen {generation} START (gap since last: {elapsed_since_last:.2f}s)")
    
    result = original_evolve(population, X, y, generation, sample_weight)
    
    gen_duration = time.time() - gen_start
    last_gen_time = time.time()
    print(f">>> Gen {generation} END (took {gen_duration:.2f}s)")
    
    return result

regressor._evolve_population = timed_evolve

try:
    pareto = regressor.fit(X, y, ['x'])
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed in {total_time:.2f}s")
    
    best = pareto.get_best()
    if best:
        print(f"Result: {best.expression}")
        print(f"MSE: {best.mse:.6e}")
except KeyboardInterrupt:
    print("\n❌ Interrupted - likely freeze detected")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
