
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("C:/Users/LOQ/PycharmProjects/kalkulator-ai"))

from kalkulator_pkg.cli.repl_commands import _detect_zeros, _detect_integer_patterns, generate_pattern_seeds

def test_rational_analysis():
    print("Testing Deductive Rational Analysis...")
    
    # 1. Test Zero Detection (Phase 2)
    # Target: f(x) = (x^3 + 1) / (x^3 - 1)
    # Zero at x = -1
    X_zero = np.array([-1.0, 0.0, 1.0]).reshape(-1, 1)
    y_zero = np.array([0.0, -1.0, np.inf]) # f(-1)=0
    
    zeros = _detect_zeros(X_zero, y_zero)
    print(f"\nPhase 2 (Zero Detection):")
    print(f"Inputs: {X_zero.flatten()}, Outputs: {y_zero}")
    print(f"Detected Zeros Seeds: {zeros}")
    
    assert "(x + 1.0)" in zeros or "(x + 1)" in zeros
    assert "(x^3 + 1)" in zeros
    print("✅ Zero detection passed")

    # 2. Test Integer Pattern Recognition (Phase 3)
    # Target: f(2) = 9/7
    X_int = np.array([2.0]).reshape(-1, 1)
    y_int = np.array([9.0/7.0])
    
    patterns = _detect_integer_patterns(X_int, y_int)
    print(f"\nPhase 3 (Integer Pattern):")
    print(f"Input: 2.0, Output: {y_int[0]}")
    print(f"Detected Patterns: {patterns}")
    
    expected_part_1 = "x^3 + 1"
    expected_part_2 = "x^3 - 1"
    
    found = False
    for p in patterns:
        p_clean = p.replace(" ", "")
        if "x^3+1" in p_clean and "x^3-1" in p_clean:
            found = True
            break
            
    if not found:
        print(f"❌ Failed to find rational pattern. Expected something containing '{expected_part_1}' and '{expected_part_2}'")
    else:
        print("✅ Integer pattern analysis passed")
    # Create a dataset for (x^3+1)/(x^3-1)
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    # Avoid singular point 1.0
    X = X[np.abs(X - 1.0) > 0.1].reshape(-1, 1)
    
    vals = X.flatten()
    y = (vals**3 + 1) / (vals**3 - 1)
    
    # Inject singularity, zero and integer point for pattern detection
    X = np.concatenate([X, np.array([[-1.0], [1.0000001], [2.0]])])
    y = np.concatenate([y, np.array([0.0, 1e9, 9.0/7.0])]) # 9/7 = (2^3+1)/(2^3-1)
    
    seeds = generate_pattern_seeds(X, y, ["x"], verbose=True)
    print(f"\nFull Integration (Generate Pattern Seeds):")
    print(f"Seeds found: {seeds}")
    
    # Check if the exact answer is in the seeds
    # It might be in various forms, but we look for key components
    has_rational = any("x^3 + 1" in s and "x^3 - 1" in s for s in seeds)
    print(f"Contains target rational structure? {has_rational}")
    
if __name__ == "__main__":
    test_rational_analysis()
