import time
import numpy as np
import warnings
import sys
from kalkulator_pkg.cli.repl_commands import _handle_evolve, generate_pattern_seeds

def test_performance():
    print("--- Setting up data for sin(cos(x)) ---")
    # Generate data
    x = np.linspace(-5, 5, 100)
    # Add complex point to trigger warnings
    x = np.concatenate([x, [1j]])
    y = np.sin(np.cos(x))
    
    # Mock data dictionary for _handle_evolve context
    # _handle_evolve parses text command. 
    # To isolate performance, we might want to call regression directly?
    # But user complained about 'alt' command flow.
    # So we should call _handle_evolve logic or simulate it.
    
    # Easier: Use the GeneticSymbolicRegressor directly with the same config as REPL
    from kalkulator_pkg.symbolic_regression import GeneticSymbolicRegressor, GeneticConfig
    
    X_train = x.reshape(-1, 1)
    y_train = y
    
    print("\n--- Testing Pattern Detection Performance ---")
    start = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seeds = generate_pattern_seeds(X_train, y_train, ['x'], verbose=True)
    pattern_time = time.time() - start
    print(f"Pattern Detection took: {pattern_time:.4f}s")
    print(f"Detected seeds: {seeds}")
    
    if w:
        print(f"Captured {len(w)} warnings during detection:")
        for warning in w:
            print(f"  {warning.category.__name__}: {warning.message}")
            
    # DIRECT SPACE
    print("\n--- Testing Direct Space Evolution ---")
    config_direct = GeneticConfig(
        population_size=300,
        generations=20, # Short run
        verbose=True,
    )
    reg = GeneticSymbolicRegressor(config_direct)
    
    start = time.time()
    # Inject seeds
    reg.population = seeds # Not exact API but effectively seeding
    # Actually need to pass seeds to fit or injection
    # In REPL: seeds.extend(auto_seeds), then injected.
    
    # We'll just run fit() and see if it finds it (simulating 'alt')
    # Use real-only data for fit to avoid crash
    mask = np.isreal(x) & np.isreal(y)
    X_real = X_train[mask]
    y_real = y_train[mask]
    
    reg.fit(X_real, y_real, seeds=seeds)
    direct_time = time.time() - start
    print(f"Direct Evolution took: {direct_time:.4f}s")
    print(f"Best program: {reg.best_program_}")
    print(f"Best MSE: {reg.min_mse_}")

    # INVERSE SPACE
    print("\n--- Testing Inverse Space Evolution (Expect Slowness?) ---")
    # y' = 1/y
    # This might have singularities if y crosses 0. 
    # sin(cos(x)) range is approx [-0.84, 0.84]. It DOES cross 0.
    # 1/sin(cos(x)) will have huge poles. 
    # This explains O(n!) feeling or instability!
    
    y_inv = 1.0 / y_real
    # Filter infs
    mask_inv = np.isfinite(y_inv)
    X_inv = X_real[mask_inv]
    y_inv_clean = y_inv[mask_inv]
    
    reg_inv = GeneticSymbolicRegressor(config_direct)
    
    start = time.time()
    try:
        reg_inv.fit(X_inv, y_inv_clean, seeds=[]) # No seeds for inverse usually
        inv_time = time.time() - start
        print(f"Inverse Evolution took: {inv_time:.4f}s")
    except Exception as e:
        print(f"Inverse Evolution Failed/Crashed: {e}")

if __name__ == "__main__":
    test_performance()
