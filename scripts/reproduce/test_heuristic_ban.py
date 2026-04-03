
import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from kalkulator_pkg.heuristics import check_power_peeling
from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

def test_heuristic_ban():
    print("Testing Heuristic Ban Enforcement...")
    
    # 1. Setup Data: y = x^0.5 (which is sqrt(x))
    # Using simple real data for simplicity in heuristic check
    # Need > 5 points to pass validation
    X = [[4], [9], [16], [25], [36], [49], [100]]
    y = [2, 3, 4, 5, 6, 7, 10]
    names = ["x"]
    
    # 2. Test without ban (Baseline)
    print("\n1. Testing without bans...")
    success, expr, mse = check_power_peeling(X, y, names, verbose=True)
    print(f"Result: {expr}")
    assert success, "Should find x^0.5 without bans"
    assert "0.5" in str(expr) or "sqrt" in str(expr)
    print("PASS: Baseline established.")
    
    # 3. Test with 'sqrt' ban (Target Scenario)
    print("\n2. Testing with 'ban sqrt'...")
    success, expr, mse = check_power_peeling(X, y, names, verbose=True, banned_operators=["sqrt"])
    print(f"Result: {expr}")
    assert not success, "Should REJECT x^0.5 when sqrt is banned"
    print("PASS: 'sqrt' ban enforced.")
    
    # 4. Test with 'pow' ban
    print("\n3. Testing with 'ban pow'...")
    success, expr, mse = check_power_peeling(X, y, names, verbose=True, banned_operators=["pow"])
    print(f"Result: {expr}")
    assert not success, "Should REJECT any power relation when pow is banned"
    print("PASS: 'pow' ban enforced.")

    # 5. Test Integration via generate_pattern_seeds
    print("\n4. Testing Integration (generate_pattern_seeds)...")
    seeds = generate_pattern_seeds(X, y, names, verbose=True, banned_operators=["sqrt"])
    print(f"Seeds found: {seeds}")
    # Should NOT contain x^0.5 or similar
    bad_seed = any("0.5" in s or "sqrt" in s for s in seeds)
    assert not bad_seed, "Seeds should not contain banned patterns"
    print("PASS: Integration verified.")

    print("\nSUCCESS: All heuristic bans verified.")

if __name__ == "__main__":
    test_heuristic_ban()
