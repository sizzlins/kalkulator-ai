import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import discover_equation, GeneticConfig

def test_floor_discovery():
    print("\n=== Testing Floor Primitive Discovery ===")
    
    # Data generation for f(x) = floor(sin(x) * 10)
    # Range similar to user's log: -20 to 20
    import kalkulator_pkg
    print(f"DEBUG: Loaded kalkulator_pkg from: {kalkulator_pkg.__file__}")
    
    X = np.linspace(-20, 20, 200)
    y = np.floor(np.sin(X) * 10)
    
    print(f"Data shape: {X.shape}")
    print(f"Sample Y: {y[:10]}")
    
    # Configure with explicit focus on finding this
    config = GeneticConfig(
        population_size=100,
        generations=20,
        verbose=True,
        # We want to see if it finds it AUTOMATICALLY without us forcing 'floor' in the config here,
        # but the user implies 'floor' isn't in the default set.
        # We will check the defaults in the code.
    )
    
    print("Running discovery...")
    pareto = discover_equation(X.reshape(-1, 1), y, config=config)
    
    best = pareto.get_best()
    print(f"\nBest Solution Found: {best.expression if best else 'None'}")
    print(f"MSE: {best.mse if best else 'N/A'}")
    
    if best and "floor" in best.expression:
        print("SUCCESS: 'floor' primitive was used!")
    else:
        print("FAILURE: 'floor' primitive NOT found in best solution.")
        # Check if it was at least mathematically close (like the user's result)
        if best and best.mse < 30 and best.mse > 0.1:
             print("Result matches user's report (approximate fit without floor).")

if __name__ == "__main__":
    test_floor_discovery()
