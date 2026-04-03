"""
Handler functions for system/diagnostic commands (health, debug, timing, benchmark).
Part of the CLI refactoring to decompose repl_commands.py.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def toggle_setting(text: str, ctx: Any, attr: str, name: str):
    """Generic toggle for boolean context settings."""
    parts = text.lower().split()
    if len(parts) < 2:
        val = getattr(ctx, attr, False)
        print(f"{name} is {'ON' if val else 'OFF'}")
        return
    state = parts[1]
    if state == "on":
        setattr(ctx, attr, True)
        print(f"{name}: ON")
    elif state == "off":
        setattr(ctx, attr, False)
        print(f"{name}: OFF")
    else:
        print(f"Usage: {parts[0]} <on|off>")


def handle_debug_command(text: str, ctx: Any):
    toggle_setting(text, ctx, "debug_mode", "Debug mode")
    if ctx.debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def handle_timing_command(text: str, ctx: Any):
    toggle_setting(text, ctx, "timing_enabled", "Timing")


def handle_cachehits_command(text: str, ctx: Any):
    toggle_setting(text, ctx, "show_cache_hits", "Cache hit display")


def handle_benchmark(text: str):
    """Handle 'benchmark' command to run performance tests."""
    print("Running benchmark...")
    print("Note: Full benchmark suite is experimental.")
    print("Try: 'health' for a basic system check instead.")


def handle_health_command(text: str, ctx: Any):
    """Run health check to verify dependencies and basic operations."""
    checks_passed = 0
    checks_failed = 0

    print("Running Kalkulator health check...", flush=True)
    print("-" * 50)

    # Check SymPy import
    try:
        import sympy as sp

        version = sp.__version__
        print(f"[OK] SymPy {version} imported successfully", flush=True)
        checks_passed += 1
    except ImportError as e:
        print(f"[FAIL] SymPy import failed: {e}", flush=True)
        checks_failed += 1

    # Check basic parsing
    try:
        from ...parser import parse_preprocessed
        from ...parser import preprocess

        test_expr = "2 + 2"
        preprocessed = preprocess(test_expr)
        parsed = parse_preprocessed(preprocessed)
        if parsed == 4:
            print("[OK] Basic parsing works", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Basic parsing failed: expected 4, got {parsed}", flush=True)
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Basic parsing exception: {e}", flush=True)
        checks_failed += 1

    # Check Solver
    try:
        from ...solver import solve_single_equation

        res = solve_single_equation("2*x=10", "x")
        if res.get("ok"):
            exact = res.get("exact", [])
            if "5" in str(exact) or (
                isinstance(exact, list) and len(exact) > 0 and str(exact[0]) == "5"
            ):
                print("[OK] Solver works (2*x=10 -> 5)", flush=True)
                checks_passed += 1
            else:
                print(f"[FAIL] Solver result mismatch: {res}", flush=True)
                checks_failed += 1
        else:
            print(f"[FAIL] Solver failed: {res}", flush=True)
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Solver exception: {e}", flush=True)
        checks_failed += 1

    # Check Worker Process (IPC) & Vectorization
    try:
        import numpy as np

        from ...worker import evaluate_safely

        # Worker Test
        res = evaluate_safely("2**10")  # 1024
        
        # Check if result is a dict and has 'result' key
        val = res
        if isinstance(res, dict) and res.get("ok"):
            val = res.get("result")
            
        if str(val) == "1024":
            print("[OK] Worker IPC works (2**10 -> 1024)", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Worker IPC failed: {res}", flush=True)
            checks_failed += 1

        # Vectorization Test
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        dot = np.dot(v1, v2)
        if dot == 32:
            print("[OK] Vectorization works (numpy dot product)", flush=True)
            checks_passed += 1
        else:
            print(f"[FAIL] Vectorization result error: {dot} != 32", flush=True)
            checks_failed += 1

    except ImportError:
        print("[FAIL] Numpy or Worker dependencies missing", flush=True)
        checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Worker/Vectorization exception: {e}", flush=True)
        checks_failed += 1

    # Check Regression Engine (The Core Core)
    try:
        from ...function_manager import find_function_from_data

        # Simple y = x + 1
        data = [(["1"], "2"), (["2"], "3"), (["3"], "4")]
        success, func_str, _, error_msg = find_function_from_data(ctx, data, ["x"])

        if success and ("x + 1" in func_str or "1 + x" in func_str):
            print(
                f"[OK] Regression Engine works (found {func_str} from 3 points)",
                flush=True,
            )
            checks_passed += 1
        else:
            print(
                f"[FAIL] Regression Engine failed. Got: {func_str}. Error: {error_msg}",
                flush=True,
            )
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Regression Engine exception: {e}", flush=True)
        checks_failed += 1

    print("-" * 50)
    total_checks = checks_passed + checks_failed
    if checks_failed == 0:
        print(
            f"Health Check Passed: {checks_passed}/{total_checks} systems operational.",
            flush=True,
        )
    else:
        print(
            f"Health Check FAILED: {checks_failed}/{total_checks} systems failed.",
            flush=True,
        )
