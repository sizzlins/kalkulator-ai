
import os
import sys
import numpy as np
from kalkulator_pkg.heuristics import detect_smoothness
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

# Mock data
# 1. Continuous (Smooth) - e.g. sin(x)
x_smooth = np.linspace(0, 10, 100).reshape(-1, 1)
y_smooth = np.sin(x_smooth)

# 2. Discrete (Stepped) - e.g. bitwise ops or quantization
x_stepped = np.linspace(0, 10, 100).reshape(-1, 1)
y_stepped = np.floor(x_stepped) # Step function

def test_smoothness_detection():
    print("Testing detect_smoothness...")
    
    # Smooth case
    is_smooth = detect_smoothness(x_smooth.tolist(), y_smooth.tolist(), verbose=True)
    if is_smooth:
        print("[PASS] Correctly identified sin(x) as smooth.")
    else:
        print("[FAIL] Incorrectly identified sin(x) as discrete!")

    # Stepped case
    is_discrete = not detect_smoothness(x_stepped.tolist(), y_stepped.tolist(), verbose=True)
    if is_discrete:
        print("[PASS] Correctly identified floor(x) as discrete.")
    else:
        print("[FAIL] Incorrectly identified floor(x) as smooth!")

def test_config_integration():
    print("\nTesting GeneticConfig integration...")
    
    # Case 1: Smooth -> Bitwise Disabled
    config_smooth = GeneticConfig(population_size=10, allow_bitwise=False)
    print(f"Smooth Config allow_bitwise: {config_smooth.allow_bitwise}")
    
    # Mocking what happens in EvolutionTrainer (we can't easily instantiate it here without full context, 
    # but we can check if the flag is carried correctly)
    
    # Case 2: Discrete -> Bitwise Allowed
    config_discrete = GeneticConfig(population_size=10, allow_bitwise=True)
    print(f"Discrete Config allow_bitwise: {config_discrete.allow_bitwise}")

if __name__ == "__main__":
    test_smoothness_detection()
    test_config_integration()
