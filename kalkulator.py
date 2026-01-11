#!/usr/bin/env python3
"""
Kalkulator: Symbolic Regression & Science Engine

Main entry point for the Kalkulator: Symbolic Regression & Science Engine application.
This file serves as a thin wrapper that delegates all functionality
to the kalkulator_pkg package.

Copyright (c) 2026 Syahbana
All rights reserved.

Usage:
    python kalkulator.py                    # Interactive REPL
    python kalkulator.py -e "2+2"           # Evaluate expression
    python kalkulator.py --help             # Show help

Terminal command to create an .EXE file using PyInstaller:
    pyinstaller --onefile --console --collect-all sympy kalkulator.py

Terminal command for Streamlit:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys

# Allow frozen executables (PyInstaller) on Windows to spawn child processes.
try:
    from multiprocessing import freeze_support
except ImportError:
    # Fallback for environments without multiprocessing
    def freeze_support() -> None:
        """No-op fallback for freeze_support."""
        pass


def main() -> int:
    """
    Main entry point for Kalkulator.

    Delegates all functionality to the kalkulator_pkg.cli module,
    which handles argument parsing, expression evaluation, and output formatting.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """

    # Load persistent cache on startup
    try:
        from kalkulator_pkg.cache_manager import load_persistent_cache

        load_persistent_cache()  # Initialize cache
    except ImportError:
        pass  # Cache manager not available, continue without persistent cache

    # Delegate to the modular CLI entrypoint
    try:
        from kalkulator_pkg.cli import main_entry

        exit_code = main_entry(sys.argv[1:])
        # Save persistent cache on normal exit
        try:
            from kalkulator_pkg.cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
        return exit_code
    except KeyboardInterrupt:
        # Stop worker processes before exiting
        try:
            from kalkulator_pkg.worker import _WORKER_MANAGER

            _WORKER_MANAGER.stop()
        except Exception:
            pass
        # Save cache on interrupt
        try:
            from kalkulator_pkg.cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
        print("\nInterrupted by user.")
        return 1
    except ImportError as e:
        print(f"Error: Failed to import kalkulator_pkg: {e}")
        print(
            "Please ensure all dependencies are installed: pip install -r requirements.txt"
        )
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


# python kalkulator.py --eval "a=1, x=a+1"
# Call freeze_support early for PyInstaller compatibility (must be at module level)
if __name__ == "__main__":
    try:
        freeze_support()
    except Exception:
        # If freeze_support fails, continue anyway
        pass
    sys.exit(main())

# f(-20), f(-19), f(-18), f(-17), f(-16), f(-15), f(-14), f(-13), f(-12), f(-11), f(-10), f(-9), f(-8), f(-7), f(-6), f(-5), f(-4), f(-3), f(-2), f(-1), f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10), f(11), f(12), f(13), f(14), f(15), f(16), f(17), f(18), f(19), f(20), f(e), f(pi), f(i), f(sin(1)), f(sin(pi)), f(4.1), f(-2.5), f(0.001), f(-0.99), f(12.345), f(-19.9), f(15.5), f(3.333), f(sqrt(2)), f(sqrt(5)), f(1/3), f(-3/4), f(2*pi), f(log(10)), f(cos(0)), f(1+2i)

# evolve --hybrid --verbose --boost 5

# f(-5, -5), f(-5, -4), f(-5, -3), f(-5, -2), f(-5, -1), f(-5, 0), f(-5, 1), f(-5, 2), f(-5, 3), f(-5, 4), f(-5, 5), f(-4, -5), f(-4, -4), f(-4, -3), f(-4, -2), f(-4, -1), f(-4, 0), f(-4, 1), f(-4, 2), f(-4, 3), f(-4, 4), f(-4, 5), f(-3, -5), f(-3, -4), f(-3, -3), f(-3, -2), f(-3, -1), f(-3, 0), f(-3, 1), f(-3, 2), f(-3, 3), f(-3, 4), f(-3, 5), f(-2, -5), f(-2, -4), f(-2, -3), f(-2, -2), f(-2, -1), f(-2, 0), f(-2, 1), f(-2, 2), f(-2, 3), f(-2, 4), f(-2, 5), f(-1, -5), f(-1, -4), f(-1, -3), f(-1, -2), f(-1, -1), f(-1, 0), f(-1, 1), f(-1, 2), f(-1, 3), f(-1, 4), f(-1, 5), f(0, -5), f(0, -4), f(0, -3), f(0, -2), f(0, -1), f(0, 0), f(0, 1), f(0, 2), f(0, 3), f(0, 4), f(0, 5), f(1, -5), f(1, -4), f(1, -3), f(1, -2), f(1, -1), f(1, 0), f(1, 1), f(1, 2), f(1, 3), f(1, 4), f(1, 5), f(2, -5), f(2, -4), f(2, -3), f(2, -2), f(2, -1), f(2, 0), f(2, 1), f(2, 2), f(2, 3), f(2, 4), f(2, 5), f(3, -5), f(3, -4), f(3, -3), f(3, -2), f(3, -1), f(3, 0), f(3, 1), f(3, 2), f(3, 3), f(3, 4), f(3, 5), f(4, -5), f(4, -4), f(4, -3), f(4, -2), f(4, -1), f(4, 0), f(4, 1), f(4, 2), f(4, 3), f(4, 4), f(4, 5), f(5, -5), f(5, -4), f(5, -3), f(5, -2), f(5, -1), f(5, 0), f(5, 1), f(5, 2), f(5, 3), f(5, 4), f(5, 5), f(4.1, 5.1), f(-2.5, 0.5), f(0.001, 0.001), f(-0.99, 0.99), f(12.34, 56.78), f(-19.9, 19.9), f(1.5, 1.5), f(3.333, 6.666), f(pi, pi), f(e, e), f(pi, e), f(e, pi), f(sqrt(2), sqrt(3)), f(sin(1), cos(1)), f(1/3, 2/3), f(-3/4, -1/4), f(2*pi, 0), f(log(10), log(2)), f(i, i), f(1+i, 1-i)