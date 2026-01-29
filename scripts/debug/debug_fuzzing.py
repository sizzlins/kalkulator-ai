
from kalkulator_pkg.parser import safe_sympy_parse, ValidationError
import sys

attacks = [
    "(1).__class__.__base__",
    "''.__class__.__mro__[1].__subclasses__()",
    "eval('1+1')",
    "__import__('os').system('ls')",
    "exec('print(1)')",
    "lambda: 1",
    "open('/etc/passwd')",
    "globals()",
    "__builtins__",
    "type.__subclasses__(type)",
]

print(f"Testing {len(attacks)} attacks...")

for i, attack in enumerate(attacks):
    try:
        print(f"[{i}] Testing: {attack}")
        safe_sympy_parse(attack)
        print(f"!!! SECURITY FAIL: Parsed successfully: {attack}")
    except ValidationError as e:
        print(f"    OK: Caught ValidationError: {e}")
    except Exception as e:
        print(f"    FAIL: Caught unexpected {type(e).__name__}: {e}")
