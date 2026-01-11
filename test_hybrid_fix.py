"""Test hybrid mode with scale-invariant fitness fix."""
import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig


def test_hybrid_mode_fixed():
    """Test that hybrid mode now works with extreme data ranges."""
    print("\n" + "="*70)
    print("TEST: Hybrid Mode with Wide Range (147 points)")
    print("="*70)
    
    # Generate same 147 points from user's command
    X = []
    y = []
    
    # Grid from -5 to 5
    for x_val in range(-5, 6):
        for y_val in range(-5, 6):
            if x_val == 0 and y_val < 0:
                continue  # Skip 0^negative
            if x_val < 0 and y_val != int(y_val):
                continue  # Skip negative ^ non-integer
            
            X.append([x_val, y_val])
            y.append(x_val ** y_val)
    
    # Add extreme values
    test_cases = [
        (-20, -19), (-18, -17), (-16, -15), (-14, -13),
        (-12, -11), (-10, -9), (-8, -7), (-6, -5),
        (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),
        (12, 13), (14, 15), (16, 17), (18, 19)
    ]
    
    for x_val, y_val in test_cases:
        X.append([x_val, y_val])
        y.append(x_val ** y_val)
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    print(f"Data points: {len(X)}")
    print(f"Y range: {y.min():.2e} to {y.max():.2e}")
    
    # Test WITH HYBRID (should now work)
    config = GeneticConfig(
        population_size=200,
        n_islands=2,
        generations=30,
        timeout=20,
        verbose=True,
        seeds=None  # Hybrid will set seeds from find()
    )
    
    regressor = GeneticSymbolicRegressor(config)
    
    # Enable hybrid mode by passing find() result as seed
    # This simulates what --hybrid does
    from kalkulator_pkg.regression_solver import solve_regression_stage
    
    print("\nRunning find() for hybrid seed...")
    success, func_str, confidence, mse_find = solve_regression_stage(
        X, y, 
        [(X[i], y[i]) for i in range(len(X))],
        ['x', 'y'],
        include_transcendentals=True
    )
    
    if success:
        print(f"find() result: {func_str[:100]}...")
        print(f"find() MSE: {mse_find:.2e}")
        
        # Use this as hybrid seed
        config.seeds = [func_str] if func_str else []
    
    print("\nRunning genetic algorithm with hybrid seed...")
    pareto = regressor.fit(X, y, variable_names=['x', 'y'])
    
    best_expr = regressor.get_expression()
    best_mse = pareto.get_best().mse if pareto.get_best() else float('inf')
    
    print(f"\n✓ Best expression: {best_expr}")
    print(f"✓ MSE: {best_mse:.2e}")
    
    # Success criteria
    success = 'x**y' in best_expr or 'pow(x' in best_expr.lower()
    if success and best_mse < 0.1:
        print("✅ SUCCESS: Hybrid mode now finds x^y with scaled seed!")
        return True
    else:
        print(f"❌ FAIL: Still having issues")
        return False


if __name__ == '__main__':
    success = test_hybrid_mode_fixed()
    
    if success:
        print("\n🎉 Hybrid mode fixed! find() now provides good seeds.")
    else:
        print("\n⚠️ Test failed. Need further investigation.")
