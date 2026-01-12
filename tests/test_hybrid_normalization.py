import numpy as np
import pytest
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig

def test_hybrid_normalization_cosh():
    # Data with large range to trigger normalization (threshold 1000)
    # cosh(10) ~ 11013
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = np.cosh(X[:, 0])
    
    # Verify range triggers normalization logic in code
    y_range = y.max() - y.min()
    print(f"Data Range: {y_range:.2f}")
    assert y_range > 1000, "Range must trigger normalization"
    
    # Configure with the CORRECT seed
    # If logic is broken, the seed (unscaled) will have huge error against normalized data
    # and will be discarded.
    config = GeneticConfig(
        population_size=50,
        generations=5, # Short run: Rely on seed survival
        seeds=["cosh(x)"],
        verbose=True
    )
    
    regressor = GeneticSymbolicRegressor(config)
    pareto = regressor.fit(X, y, variable_names=["x"])
    
    best = pareto.get_best()
    if best is None:
        pytest.fail("No solution found")
        
    print(f"\nBest Expression: {best.expression}")
    print(f"Best MSE reported: {best.mse}")
    
    # 1. Check if seed survived (contains cosh)
    assert "cosh" in best.expression, "Seed 'cosh(x)' was lost/discarded!"
    
    # 2. Check denormalization
    # The expression should NOT look like (cosh(x)-1)/11000
    # It should look like cosh(x)
    # With simplification, it should be clean.
    
    # 3. Check actual prediction accuracy (Denormalization must be correct)
    preds = regressor.predict(X)
    real_mse = np.mean((preds - y)**2)
    print(f"Validation MSE: {real_mse}")
    
    # Tolerance: Floating point operations might introduce small error
    # but it should be very small compared to range (1e4)
    assert real_mse < 1e-1, f"MSE too high ({real_mse}). Denormalization failed?"

if __name__ == "__main__":
    test_hybrid_normalization_cosh()
