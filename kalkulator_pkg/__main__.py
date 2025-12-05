"""Main entry point for running kalkulator_pkg as a module.

This allows running Kalkulator with:
    python -m kalkulator_pkg
    python -m kalkulator_pkg --health-check
    python -m kalkulator_pkg -e "2+2"

This is equivalent to running:
    python -m kalkulator_pkg.cli
    python kalkulator.py
"""

from __future__ import annotations

import sys

from .cli import main_entry

if __name__ == "__main__":
    sys.exit(main_entry())
