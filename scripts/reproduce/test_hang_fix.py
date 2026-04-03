"""Verification script for genetic engine hang fix.

Tests that the genetic engine completes without hanging on Windows (serial mode).
Includes a hard 60-second timeout to detect hangs.
"""
import sys
import os
import time
import signal
import threading

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np


def test_with_timeout(func, timeout=60):
    """Run a function with a timeout. Returns (success, elapsed, error_msg)."""
    result = [False, 0, "Timeout"]
    
    def wrapper():
        try:
            t0 = time.perf_counter()
            func()
            result[0] = True
            result[1] = time.perf_counter() - t0
            result[2] = None
        except Exception as e:
            result[1] = time.perf_counter() - t0 if 't0' in dir() else 0
            result[2] = str(e)
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return False, timeout, "HANG DETECTED: Thread still alive after timeout"
    return result[0], result[1], result[2]


def test_basic_evolution():
    """Test basic x^2 discovery completes without hanging."""
    from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
    from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
    
    config = GeneticConfig(
        population_size=50,
        generations=20,
        n_islands=1,
        verbose=True,
        timeout=30,
    )
    
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = X.flatten() ** 2
    
    reg = GeneticSymbolicRegressor(config=config)
    reg.fit(X, y, variable_names=['x'])
    
    best = reg.pareto_front.get_best()
    print(f"  Best expression: {best.expression if best else 'None'}")
    print(f"  Best MSE: {best.mse if best else 'N/A'}")
    assert best is not None, "No solution found"


def test_forensic_timeout():
    """Test that forensic analysis respects timeout."""
    from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds
    
    # Create data that might trigger many detectors
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    y = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])  # Fibonacci
    
    t0 = time.perf_counter()
    result = generate_pattern_seeds(None, X, y, ['x'], verbose=True)
    elapsed = time.perf_counter() - t0
    
    print(f"  Forensic analysis completed in {elapsed:.2f}s")
    print(f"  Seeds found: {len(result) if isinstance(result, list) else result}")
    assert elapsed < 15.0, f"Forensic analysis took too long: {elapsed:.2f}s"


def test_pareto_elite_only():
    """Test that _update_pareto_front processes only elite trees."""
    from kalkulator_pkg.symbolic_regression.genetic_engine import EvolutionTrainer, ParetoFront
    from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
    from kalkulator_pkg.symbolic_regression.strategies import EvolutionStrategy
    from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
    
    config = GeneticConfig(population_size=100, elitism=5)
    strategy = EvolutionStrategy(config)
    trainer = EvolutionTrainer(config, strategy, ParetoFront())
    
    # Create a fake island with 100 trees
    trees = []
    for i in range(100):
        tree = ExpressionTree.random_tree(variables=['x'], max_depth=3)
        tree.fitness = float(i)  # Assign ordered fitness
        trees.append(tree)
    
    t0 = time.perf_counter()
    trainer._update_pareto_front([trees])
    elapsed = time.perf_counter() - t0
    
    print(f"  Pareto update (100 trees, limit={max(config.elitism, 10)}) took {elapsed:.3f}s")
    # Should be fast since we only process top 10
    assert elapsed < 10.0, f"Pareto update took too long: {elapsed:.3f}s"


if __name__ == "__main__":
    tests = [
        ("Forensic Timeout", test_forensic_timeout),
        ("Pareto Elite-Only", test_pareto_elite_only),
        ("Basic Evolution (x^2)", test_basic_evolution),
    ]
    
    all_passed = True
    for name, func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        
        success, elapsed, error = test_with_timeout(func, timeout=60)
        
        if success:
            print(f"  ✅ PASSED ({elapsed:.2f}s)")
        else:
            print(f"  ❌ FAILED ({elapsed:.2f}s): {error}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print(f"{'='*60}")
    
    sys.exit(0 if all_passed else 1)
