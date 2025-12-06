#!/usr/bin/env python3
"""
Kalkulator: Symbolic Regression & Science Engine

Main entry point for the Kalkulator: Symbolic Regression & Science Engine application.
This file serves as a thin wrapper that delegates all functionality
to the kalkulator_pkg package.

Copyright (c) 2025 Syahbana
All rights reserved.

Usage:
    python kalkulator.py                    # Interactive REPL
    python kalkulator.py -e "2+2"           # Evaluate expression
    python kalkulator.py --help             # Show help

For PyInstaller:
    pyinstaller --onefile --console --collect-all sympy kalkulator.py
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


# >>> f(312321) = 624642, f(pi) = 6.28318530717958, find f(x)
# f(x) = 6.282458077*exp(3.684405675e-05*x)
# Function 'f' is now available. You can call it like: f(values)
# >>> f(312321) = 624642, f(pi) = 6.28318530717958, f(5) = 10,find f(x)
# f(x) = 2*x^1
# Function 'f' is now available. You can call it like: f(values)
