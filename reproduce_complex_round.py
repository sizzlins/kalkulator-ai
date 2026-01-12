
import numpy as np
from kalkulator_pkg.utils.formatting import format_number, format_solution

def test_round_complex():
    c = np.complex128(1.2 + 3.4j)
    try:
        # Simulate format_number behavior or similar
        print(f"Trying to round: {c}")
        # Direct round on complex128 throws the error
        r = round(c)
        print(f"Rounded: {r}")
    except TypeError as e:
        print(f"Caught expected error: {e}")

    try:
        # Simulate usage in formatting.py
        # Lines 660: if abs(val_f - round(val_f)) < 1e-9:
        # If val_f is complex, round(val_f) fails.
        val_f = c
        r = round(val_f)
    except TypeError as e:
         print(f"Caught error in simulated formatting check: {e}")

if __name__ == "__main__":
    test_round_complex()
