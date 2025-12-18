#!/usr/bin/env python3
"""Test single-parameter function finding."""

import traceback

from kalkulator_pkg.function_manager import find_function_from_data

try:
    result = find_function_from_data(
        [([2], 12.5664), ([100.64], 31819.3103), ([150], 70685.7750)],
        ['x']
    )
    print('Success:', result[0])
    if result[0]:
        print('Function:', result[1])
    else:
        print('Error:', result[3])
except Exception as e:
    print(f'Exception: {e}')
    traceback.print_exc()

