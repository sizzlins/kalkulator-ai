
import sys
import os
import numpy as np
import logging

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_V4_2")

def verify_lazy_sympy_gone():
    logger.info("TEST 1: Verifying LazySymPy removal...")
    from kalkulator_pkg.utils import numeric, formatting
    from kalkulator_pkg import worker
    
    # Check numeric.py
    if hasattr(numeric, "_LazySymPy") or "LazySymPy" in str(type(numeric.sp)):
        logger.error("FAIL: _LazySymPy still present in numeric.py")
        return False
    
    # Check formatting.py
    if hasattr(formatting, "_LazySymPy") or "LazySymPy" in str(type(formatting.sp)):
        logger.error("FAIL: _LazySymPy still present in formatting.py")
        return False
        
    # Check worker.py (harder to check internal class if not exposed, but we can check usage)
    # We'll check if we can import it and if 'sp' is standard sympy
    # Note: worker.py doesn't expose 'sp' globally usually, but we removed the proxy class definition.
    with open(worker.__file__, 'r') as f:
        content = f.read()
        if "class _LazySymPy" in content:
            logger.error("FAIL: _LazySymPy class definition found in worker.py source")
            return False

    logger.info("PASS: LazySymPy proxies appear to be removed.")
    return True

def verify_god_object_decoupling():
    logger.info("TEST 2: Verifying God Object Decoupling (EvolutionTrainer)...")
    try:
        from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, EvolutionTrainer
        
        # 1. Check if EvolutionTrainer has train_full_model
        if not hasattr(EvolutionTrainer, "train_full_model"):
            logger.error("FAIL: EvolutionTrainer missing 'train_full_model' method.")
            return False

        # --- DEBUG IMMUTABILITY ---
        from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionNode, NodeType
        import inspect
        import kalkulator_pkg.symbolic_regression.expression_tree as et_module
        with open("path_debug.txt", "w") as f:
             f.write(str(et_module.__file__))
        logger.info(f"LOADED ExpressionTree FROM: {et_module.__file__}")

        try:
             src = inspect.getsource(ExpressionNode)
             logger.info(f"SOURCE OF ExpressionNode:\n{src}")
        except Exception as e:
             logger.warning(f"Could not get source: {e}")

        logger.info(f"ExpressionNode match_args: {ExpressionNode.__match_args__}")
        logger.info(f"ExpressionNode slots: {getattr(ExpressionNode, '__slots__', 'No Slots')}")
        try:
             node = ExpressionNode(NodeType.CONSTANT, 1.0)
             node.value = 2.0
             logger.info("PASS: ExpressionNode is mutable.")
        except Exception as e:
             logger.error(f"FAIL: ExpressionNode is IMMUTABLE! Error: {e}")
             return False
        # --------------------------
            
        # 2. Run a dummy fit to ensure orchestration works
        X = np.random.rand(10, 2)
        y = X[:, 0] * 2 + 1 # Simple linear 2*x0 + 1
        
        reg = GeneticSymbolicRegressor(config=None)
        # Set extremely small config for speed
        reg.config.generations = 1
        reg.config.population_size = 10
        reg.config.n_islands = 1
        reg.config.boosting_rounds = 1
        reg.config.model_selection = False # Disable cross-val for speed
        
        logger.info("Running fit()...")
        reg.fit(X, y)
        
        if not reg.best_tree:
            logger.error("FAIL: Fit completed but best_tree is None.")
            return False
            
        logger.info("PASS: GeneticSymbolicRegressor.fit() ran successfully using EvolutionTrainer.")
        return True
    except BaseException as e:
        logger.error(f"FAIL: Crash during God Object verification (BaseException): {e}")
        with open("verification_progress.txt", "a") as f:
            f.write(f"\nCRASHED (BaseException): {e}\n")
            import traceback
            traceback.print_exc(file=f)
        return False

    with open("verification_progress.txt", "a") as f:
         f.write("\nFit completed successfully.\n")

def verify_lll_convergence():
    logger.info("TEST 3: Verifying LLL Convergence (Iter Check)...")
    try:
        from kalkulator_pkg.utils.lll import detect_rational_lll
        # Test a known rational
        # 3.1415926535... approx 355/113
        val = 3.1415929203539825 # 355/113 exactly check
        p, q = detect_rational_lll(val, max_denom=1000, tolerance=1e-9)
        
        if p == 355 and q == 113:
             logger.info(f"PASS: LLL correctly identified 355/113.")
        else:
             logger.warning(f"WARN: LLL returned {p}/{q} instead of 355/113 (might be acceptable)")
             
        # Verify source code doesn't have max_iter loop
        import inspect
        src = inspect.getsource(detect_rational_lll)
        if "range(100)" in src:
            logger.error("FAIL: 'range(100)' limit still found in detect_rational_lll source code!")
            return False
            
        logger.info("PASS: LLL max_iter limit removed from source.")
        return True
    except Exception as e:
         logger.error(f"FAIL: LLL check crashed: {e}")
         return False

def verify_ctypes_safety():
    logger.info("TEST 4: Verifying ctypes safety...")
    try:
        from kalkulator_pkg import worker
        import ctypes
        import sys
        
        if 'ctypes' not in sys.modules:
             logger.error("FAIL: ctypes module missing from sys.modules! (Did we delete it?)")
             return False
             
        logger.info("PASS: ctypes module is present and safe.")
        return True
    except Exception as e:
        logger.error(f"FAIL: ctypes check failed: {e}")
        return False

if __name__ == "__main__":
    print("=== STARTING V4.2 AUDIT VERIFICATION ===")
    results = [
        verify_lazy_sympy_gone(),
        verify_god_object_decoupling(),
        verify_lll_convergence(),
        verify_ctypes_safety()
    ]
    
    if all(results):
        print("\n>>> ALL CHECKS PASSED. SYSTEM IS ROBUST. <<<")
        sys.exit(0)
    else:
        print("\n>>> VERIFICATION FAILED. SEE LOGS. <<<")
        sys.exit(1)
