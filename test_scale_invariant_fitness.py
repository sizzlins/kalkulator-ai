"""Test scale-invariant fitness function for power functions."""
import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig


def test_power_function_wide_range():
    """Test x^y discovery with extreme value range (original failure case)."""
    print("\n" + "="*70)
    print("TEST 1: Power Function with Wide Range (148 points, 1e-25 to 1e23)")
    print("="*70)
    
    # Generate x^y data with extreme range (original failure case)
    X = []
    y = []
    
    # Grid from -5 to 5
    for x_val in range(-5, 6):
        for y_val in range(-5, 6):
            if x_val == 0 and y_val < 0:
                continue  # Skip 0^negative (infinity)
            if x_val < 0 and y_val != int(y_val):
                continue  # Skip negative base with non-integer exponent
            
            X.append([x_val, y_val])
            y.append(x_val ** y_val)
    
    # Add some larger values
    test_cases = [
        (-20, -19), (-18, -17), (-16, -15), (-14, -13), 
        (-12, -11), (-10, -9), (-8, -7), (-6, -5),
        (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)
    ]
    
    for x_val, y_val in test_cases:
        X.append([x_val, y_val])
        y.append(x_val ** y_val)
    
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    print(f"Data points: {len(X)}")
    print(f"Y range: {y.min():.2e} to {y.max():.2e}")
    print(f"Orders of magnitude: {np.log10(y.max()) - np.log10(abs(y.min())):.1f}")
    
    # Configure genetic algorithm
    config = GeneticConfig(
        population_size=200,
        n_islands=2,
        generations=30,
        timeout=20,
        verbose=True,
        operators=['add', 'sub', 'mul', 'div', 'pow', 'exp', 'log']
    )
    
    regressor = GeneticSymbolicRegressor(config)
    print("\nRunning genetic algorithm...")
    pareto = regressor.fit(X, y, variable_names=['x', 'y'])
    
    best_expr = regressor.get_expression()
    best_mse = pareto.get_best().mse if pareto.get_best() else float('inf')
    
    print(f"\n✓ Best expression: {best_expr}")
    print(f"✓ MSE: {best_mse:.2e}")
    
    # Check if it found the right pattern
    success = 'x**y' in best_expr or 'pow(x' in best_expr.lower()
    if success:
        print("✅ SUCCESS: Found x^y pattern!")
    else:
        print(f"❌ FAIL: Expected x^y but got {best_expr}")
    
    return success


def test_power_function_clean_data():
    """Test x^y discovery with clean 6-point data (should still work perfectly)."""
    print("\n" + "="*70)
    print("TEST 2: Power Function with Clean Data (6 points, 1 to 100)")
    print("="*70)
    
    # Clean 6-point data that worked before
    data = [
        (2, 2, 4),
        (2, 3, 8),
        (3, 2, 9),
        (4, 2, 16),
        (2, 0, 1),
        (10, 2, 100)
    ]
    
    X = np.array([[x, y] for x, y, _ in data], dtype=float)
    y = np.array([z for _, _, z in data], dtype=float)
    
    print(f"Data points: {len(X)}")
    print(f"Y range: {y.min():.1f} to {y.max():.1f}")
    
    config = GeneticConfig(
        population_size=100,
        n_islands=2,
        generations=20,
        timeout=10,
        verbose=True,
    )
    
    regressor = GeneticSymbolicRegressor(config)
    print("\nRunning genetic algorithm...")
    pareto = regressor.fit(X, y, variable_names=['x', 'y'])
    
    best_expr = regressor.get_expression()
    best_mse = pareto.get_best().mse if pareto.get_best() else float('inf')
    
    print(f"\n✓ Best expression: {best_expr}")
    print(f"✓ MSE: {best_mse:.2e}")
    
    # Should find it quickly with low MSE
    success = best_mse < 1e-6 and ('x**y' in best_expr or 'pow(x' in best_expr.lower())
    if success:
        print("✅ SUCCESS: Still finds x^y with clean data!")
    else:
        print(f"⚠️ REGRESSION: Clean data test degraded")
    
    return success


if __name__ == '__main__':
    results = []
    
    # Run tests in sequence
    results.append(("Wide Range Test", test_power_function_wide_range()))
    results.append(("Clean Data Test", test_power_function_clean_data()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed! Scale-invariant fitness is working!")
    else:
        print("\n⚠️ Some tests failed. Review output above.")
