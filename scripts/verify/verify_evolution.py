
import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from kalkulator_pkg.symbolic_regression.genetic_engine import discover_equation
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

def test_factorial_fix():
    print("\n=== Testing Factorial Fix and Multi-Space Sanity Check ===")
    
    # User Dataset (Reduced to finite domain for clear verification)
    # Exclude x=-1 because f(-1)=inf and fitting infinity yields NaN errors in standard MSE
    X = np.array([0, 1, 2, 3, 4, 5])
    # y = x^3 + x!
    y = np.array([1, 2, 10, 33, 88, 245])
    
    print(f"Data: X={X}, y={y}")
    
    # Configure with high verbosity to see "Multiplex" logs
    config = GeneticConfig(
        population_size=100, 
        generations=20, 
        verbose=True,
        # Ensure factorial is allowed (it should be default now)
    )
    
    print("\nRunning discovery...")
    pareto = discover_equation(X.reshape(-1, 1), y, config=config)
    
    best = pareto.get_best()
    print(f"\nBest Solution Found: {best.expression if best else 'None'}")
    print(f"MSE: {best.mse if best else 'N/A'}")
    
    if best and "factorial" in best.expression:
        print("SUCCESS: 'factorial' primitive was used!")
    elif best and "gamma" in best.expression:
        print("SUCCESS: 'gamma' primitive was used!")
    else:
        print("FAILURE: Factorial/Gamma not found in best solution.")

    # Check for hallucination
    # If MSE is huge (> 1000) and it claims success, that's a failure of the sanity check.
    if best and best.mse > 1000:
        print("FAILURE: Sanity check failed? High MSE result returned.")
    elif best and best.mse < 1e-9:
        print("SUCCESS: Perfect fit found.")

if __name__ == "__main__":
    test_factorial_fix()
