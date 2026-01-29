"""Arity Check Test for Gene Bank.

Tests:
1. TEACH: Discover x + y → saves v0 + v1
2. TRANSFER: Discover a + b → injects a+b AND b+a (permutations)
3. FAIL-SAFE: Discover single-var t → should NOT inject 2-var gene
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank

def main():
    print("=" * 60)
    print("ARITY CHECK TEST - Permutation Guardrails")
    print("=" * 60)
    
    # Clear Gene Bank for clean test
    bank = get_gene_bank()
    bank.clear()
    print(f"\n[Setup] Gene Bank cleared.")
    
    config = GeneticConfig(
        population_size=50,
        generations=10,
        verbose=True,
        early_stop_mse=1e-6,
        timeout=20
    )
    
    # --- TEST 1: TEACH - Discover x + y ---
    print("\n" + "-" * 60)
    print("TEST 1: TEACH - Discovering f(x,y) = x + y")
    print("-" * 60)
    
    np.random.seed(42)
    X1 = np.random.uniform(-5, 5, (30, 2))
    y1 = X1[:, 0] + X1[:, 1]
    
    reg1 = GeneticSymbolicRegressor(config)
    reg1.fit(X1, y1, variable_names=['x', 'y'])
    
    print(f"\n[Test 1] Discovered: {reg1.get_expression()}")
    
    bank = get_gene_bank()
    print(f"\n[Gene Bank] After Test 1:")
    for g in bank.list_genes():
        print(f"  [{g['id']}] {g['expression']} (vars={g['n_vars']})")
    
    if not bank.genes:
        print("  (empty - may not have met quality threshold)")
    
    # --- TEST 2: TRANSFER - Discover a + b ---
    print("\n" + "-" * 60)
    print("TEST 2: TRANSFER - Discovering g(a,b) = a + b")
    print("Expected: Should inject permutations (a+b, b+a)")
    print("-" * 60)
    
    np.random.seed(123)
    X2 = np.random.uniform(-5, 5, (30, 2))
    y2 = X2[:, 0] + X2[:, 1]
    
    reg2 = GeneticSymbolicRegressor(config)
    reg2.fit(X2, y2, variable_names=['a', 'b'])  # Different variable names!
    
    print(f"\n[Test 2] Discovered: {reg2.get_expression()}")
    
    # --- TEST 3: FAIL-SAFE - Discover single-var h(t) = t ---
    print("\n" + "-" * 60)
    print("TEST 3: FAIL-SAFE - Discovering h(t) = t (1 variable)")
    print("Expected: Should NOT inject 2-var gene (v0 + v1)")
    print("-" * 60)
    
    np.random.seed(456)
    X3 = np.linspace(-5, 5, 30).reshape(-1, 1)
    y3 = X3.flatten()
    
    reg3 = GeneticSymbolicRegressor(config)
    reg3.fit(X3, y3, variable_names=['t'])  # Only 1 variable!
    
    print(f"\n[Test 3] Discovered: {reg3.get_expression()}")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
TEST 1: Did it save 'v0 + v1' (normalized)?
TEST 2: Did it inject learned gene with variable mapping?
TEST 3: Did it correctly SKIP the 2-var gene for 1-var problem?

If you saw:
- TEST 2: "[GeneBank] Injected N learned genes" → PASS
- TEST 3: NO "[GeneBank] Injected" message → PASS (Arity Filter works!)
""")

if __name__ == "__main__":
    main()
