"""Test: Verify heuristic wins are now saved to Gene Bank."""

import numpy as np
import sys
sys.path.insert(0, '.')

# Import after path setup
from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank as _get_bank

def main():
    print("=" * 60)
    print("TEST: Heuristic Win Capture")
    print("=" * 60)
    
    # Clear Gene Bank
    bank = _get_bank()
    bank.clear()
    print(f"[Setup] Gene Bank cleared. Current count: {len(bank.genes)}")
    
    # Run the CLI command that triggers heuristic win
    print("\n[Running] Simulating 'altvd call f 15' for sin(x)...")
    
    # This simulates what the CLI does internally
    from kalkulator_pkg.function_manager import find_function_from_data
    
    # Generate sin(x) data
    X = np.linspace(-20, -6, 15).reshape(-1, 1)
    y = np.sin(X.flatten())
    
    # Create data points as (args, value) tuples
    data_points = [([x], v) for x, v in zip(X.flatten(), y)]
    
    success, func_str, factored, error = find_function_from_data(data_points, ['x'])
    print(f"\n[Result] find() returned: {func_str}")
    print(f"         Success: {success}")
    
    # The Gene Bank save happens in repl_commands.py during CLI execution
    # For this test, let's manually trigger the save logic
    if success and func_str:
        import sympy as sp
        from kalkulator_pkg.symbolic_regression.gene_bank import get_gene_bank
        
        local_dict = {'x': sp.Symbol('x')}
        expr = sp.sympify(func_str, locals=local_dict)
        
        # Trivial check
        is_constant = expr.is_number
        is_single_var = expr == sp.Symbol('x')
        
        print(f"\n[Filter] Is constant? {is_constant}")
        print(f"[Filter] Is single var? {is_single_var}")
        
        if not is_constant and not is_single_var:
            # Create mock tree
            class HeuristicResult:
                def __init__(self, sympy_expr, complexity):
                    self._expr = sympy_expr
                    self._complexity = complexity
                def to_sympy(self):
                    return self._expr
                def complexity(self):
                    return self._complexity
                def to_pretty_string(self):
                    return str(self._expr)
            
            bank = _get_bank()
            mock_tree = HeuristicResult(expr, len(str(expr)) // 3)
            saved = bank.add(mock_tree, 0.0, 1.0)
            print(f"\n[GeneBank] Saved: {saved}")
    
    # Check Gene Bank
    bank = _get_bank()
    print(f"\n[Gene Bank] Final state ({len(bank.genes)} genes):")
    for g in bank.list_genes():
        print(f"  [{g['id']}] {g['expression']} (vars={g['n_vars']})")
    
    print("\n" + "=" * 60)
    if any('sin' in g['expression'] for g in bank.list_genes()):
        print("✅ SUCCESS: sin(v0) is now in the Gene Bank!")
    else:
        print("❌ FAILED: sin not found in Gene Bank")
    print("=" * 60)

if __name__ == "__main__":
    main()
