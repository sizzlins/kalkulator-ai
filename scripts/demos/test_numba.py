"""Verification script for Numba Evaluator.

Checks:
1. Is Numba installed and active?
2. Does evaluate_rpn_fast match NumPy ground truth?
3. Benchmarks performance (Numba vs Python fallback).
"""

import time
import numpy as np
import sys
sys.path.insert(0, '.')

from kalkulator_pkg.symbolic_regression.numba_evaluator import (
    evaluate_rpn_fast, compile_rpn_numba, HAS_NUMBA,
    OP_VAR, OP_ADD, OP_MUL, OP_SIN, OP_CONST
)

def main():
    print("=" * 60)
    print("NUMBA EVALUATOR VERIFICATION")
    print("=" * 60)
    
    print(f"Numba Installed: {HAS_NUMBA}")
    
    # Define expression: x^2 + sin(y)
    # RPN: x, x, MUL, y, SIN, ADD
    # Opcodes: OP_VAR(0), OP_VAR(0), OP_MUL, OP_VAR(1), OP_SIN, OP_ADD
    
    print("\n[Test] Expression: x^2 + sin(y)")
    
    # 1. Setup Data
    N = 1_000_000
    print(f"[Setup] Generating {N:,} points...")
    X = np.random.uniform(-10, 10, (N, 2))
    
    # 2. Ground Truth
    start = time.time()
    y_true = X[:, 0]**2 + np.sin(X[:, 1])
    t_numpy = time.time() - start
    print(f"[Numpy] Time: {t_numpy:.4f}s")
    
    # 3. Compile RPN
    # Token format for compile_rpn_numba: (type_code, val)
    # Types: 'VAR', 'CONST', 'UNARY', 'BINARY'
    # But compile_rpn_numba expects raw tokens from parser?
    # Let's manually verify compile_rpn_numba logic or construct opcodes directly.
    # The file shows compile_rpn_numba takes raw_tokens list.
    
    raw_tokens = [
        ('VAR', 0),
        ('VAR', 0),
        ('BINARY', 'mul'),
        ('VAR', 1),
        ('UNARY', 'sin'),
        ('BINARY', 'add')
    ]
    
    # Mocking var_map
    var_map = {0: 0, 1: 1} # Identity mapping for raw indices
    
    # Actually, var_map maps "x" -> 0. Let's fix raw tokens.
    # compile_rpn_numba logic:
    # if type_code == 'VAR': values.append(var_map.get(val, 0))
    # So if val is 0, var_map key must be 0? usually strings.
    
    # Let's bypass compile helper and create arrays directly to test core evaluator
    opcodes = np.array([
        OP_VAR, OP_VAR, OP_MUL, 
        OP_VAR, OP_SIN, OP_ADD
    ], dtype=np.int32)
    
    values = np.array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    ], dtype=np.float64)
    
    # 4. Evaluate Numba/Fast
    start = time.time()
    # First run triggers JIT compilation
    y_pred = evaluate_rpn_fast(opcodes, values, X)
    t_jit = time.time() - start
    print(f"[Numba] First Run (JIT + Exec): {t_jit:.4f}s")
    
    # Second run (Warm)
    start = time.time()
    y_pred_warm = evaluate_rpn_fast(opcodes, values, X)
    t_warm = time.time() - start
    print(f"[Numba] Warm Run: {t_warm:.4f}s")
    
    # 5. Check Accuracy
    mse = np.mean((y_true - y_pred_warm)**2)
    print(f"\n[Accuracy] MSE: {mse:.6e}")
    if mse < 1e-10:
        print("✅ SUCCESS: Results match NumPy precision.")
    else:
        print("❌ FAIL: Results diverge significantly.")

    # 6. Fallback Check (Force Python)
    from kalkulator_pkg.symbolic_regression.numba_evaluator import _evaluate_rpn_python
    start = time.time()
    y_py = _evaluate_rpn_python(opcodes, values, X)
    t_py = time.time() - start
    print(f"[Python] Fallback Time: {t_py:.4f}s")
    
    mse_py = np.mean((y_true - y_py)**2)
    if mse_py < 1e-10:
         print("✅ SUCCESS: Python Fallback matches NumPy.")
    else:
         print("❌ FAIL: Python Fallback results diverge.")

    if HAS_NUMBA:
        speedup = t_py / t_warm
        print(f"\n[Perf] Numba Speedup vs Python Fallback: {speedup:.1f}x")
    else:
        print("\n[Perf] Numba not installed, running in pure Python mode.")

if __name__ == "__main__":
    main()
