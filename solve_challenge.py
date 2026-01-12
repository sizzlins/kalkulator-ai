
import numpy as np
import warnings
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

warnings.filterwarnings('ignore')

def solve_challenge():
    print("\n=== Solving User Challenge: x^y on [-5, 5] Grid ===")
    
    # Generate the 11x11 grid
    X_list = []
    y_list = []
    
    range_vals = list(range(-5, 6)) # -5 to 5
    
    for x in range_vals:
        for y_val in range_vals:
            X_list.append([x, y_val])
            # Calculate x^y (complex safe)
            # numpy.lib.scimath.power handles negative base correctly
            val = np.lib.scimath.power(x, y_val)
            y_list.append(val)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Dataset: {len(X)} points")
    print(f"Sample data:")
    for i in range(5):
        print(f"  f({X[i,0]}, {X[i,1]}) = {y[i]}")
        
    config = GeneticConfig(
        population_size=1000,
        generations=20,
        parsimony_coefficient=0.01,
        verbose=True
    )
    
    reg = GeneticSymbolicRegressor(config)
    
    print("\nEvolving...")
    try:
        reg.fit(X, y, variable_names=["x", "y"])
        print(f"\n[SUCCESS] Result: {reg.best_tree.expression}")
        print(f"MSE: {reg.best_tree.fitness}")
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    solve_challenge()
