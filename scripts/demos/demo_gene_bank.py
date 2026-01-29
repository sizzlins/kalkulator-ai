"""Live demonstration of Gene Bank (Meta-Learning).

This script:
1. Discovers sin(x) from data
2. Shows the Gene Bank has learned it
3. Runs a second discovery to show recall/injection
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank

def main():
    print("=" * 60)
    print("GENE BANK LIVE DEMONSTRATION")
    print("=" * 60)
    
    # Clear Gene Bank for clean demo
    bank = get_gene_bank()
    bank.clear()
    print(f"\n[Setup] Gene Bank cleared. Current genes: {len(bank.genes)}")
    
    # --- Session 1: Discover sin(x) ---
    print("\n" + "-" * 60)
    print("SESSION 1: Discovering sin(x)")
    print("-" * 60)
    
    np.random.seed(42)
    X1 = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
    y1 = np.sin(X1.flatten())
    
    config = GeneticConfig(
        population_size=50,
        generations=20,
        verbose=True,
        early_stop_mse=1e-6,
        timeout=30
    )
    
    reg = GeneticSymbolicRegressor(config)
    reg.fit(X1, y1, variable_names=['x'])
    
    expr = reg.get_expression()
    print(f"\n[Session 1] Discovered: {expr}")
    
    # Check Gene Bank
    bank = get_gene_bank()  # Refresh
    print(f"\n[Gene Bank] After Session 1:")
    for g in bank.list_genes():
        print(f"  [{g['id']}] {g['expression']} (vars={g['n_vars']}, complexity={g['complexity']})")
    
    if not bank.genes:
        print("  (empty - expression may not have met quality threshold)")
    
    # --- Session 2: Discover sin(t) ---
    print("\n" + "-" * 60)
    print("SESSION 2: Discovering sin(t) (should inject learned gene)")
    print("-" * 60)
    
    np.random.seed(123)
    X2 = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
    y2 = np.sin(X2.flatten())
    
    reg2 = GeneticSymbolicRegressor(config)
    reg2.fit(X2, y2, variable_names=['t'])  # Different variable name!
    
    expr2 = reg2.get_expression()
    print(f"\n[Session 2] Discovered: {expr2}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nIf you see '[GeneBank] Injected N learned genes from memory.'")
    print("in Session 2, the meta-learning is working!")

if __name__ == "__main__":
    main()
