
import numpy as np
import sys
import os
import time
import types

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_freeze_with_timer():
    print("Testing for O(n!) freeze behavior with sin(1/(x-3)) data...")
    
    # User's data (excluding the filtered/complex ones based on repl log)
    # The REPL log shows valid evaluations for these points.
    X_list = [
        4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1,
        # 3.0 is nan, skipped
        2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 
        1.4, 1.3, 1.2, 1.1, 1.0,
        -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, 
        -5, -4, -3, -2, -1, 0, 1, 2, 
        # 3 skipped
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        # e, pi, sin(1), sin(pi) - approximate them
        np.e, np.pi, np.sin(1), np.sin(np.pi),
        4.1, -2.5, 0.001, -0.99, 12.345, -19.9, 15.5, 3.333,
        np.sqrt(2), np.sqrt(5), 1/3, -3/4, 2*np.pi, np.log(10), np.cos(0)
    ]
    
    # Calculate y values based on f(x) from user input (or re-calculate targets)
    # The user provided y values in the prompt. Let's use the explicit ones where possible
    # or just use the target function sin(1/(x-3)) which generated them.
    # The user Prompt: "f(x)=sin(1/(x-3))"
    X = np.array(X_list).reshape(-1, 1)
    # Recalculate y to be safe and match precision
    y = np.sin(1/(X - 3))
    
    # Convert y to 1D
    y = y.reshape(-1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Configure Regressor
    config = GeneticConfig(
        population_size=300, 
        generations=20, # Short run to test early gens
        verbose=True,
        # mimic REPL boosting
        boosting_rounds=1
    )
    
    reg = GeneticSymbolicRegressor(config)
    
    # Monkey patch _evolve_population to time it
    original_evolve = reg._evolve_population
    
    def timed_evolve(self, population, X, y, generation, sample_weight=None):
        start_time = time.time()
        print(f"DEBUG: Starting Generation {generation}...", end="", flush=True)
        
        new_pop = original_evolve(population, X, y, generation, sample_weight)
        
        elapsed = time.time() - start_time
        print(f" DONE in {elapsed:.4f}s")
        return new_pop
    
    # Bind the monkey patch
    reg._evolve_population = types.MethodType(timed_evolve, reg)
    
    print("\nStarting evolution loop...")
    start_total = time.time()
    
    try:
        reg.fit(X, y, ['x'])
    except KeyboardInterrupt:
        print("\nInterrupted by user/script timeout.")
    
    print(f"\nTotal time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    test_freeze_with_timer()
