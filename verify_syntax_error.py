
import sys
import os
import sympy as sp
from typing import Any

# Simulation of evaluate_safely + REPL handling
def evaluate_safely(expression: str, allowed_functions=None) -> dict[str, Any]:
    try:
        # Mocking sympify failure for print("Hello world")
        if "print" in expression:
             # Basic sympify fails on this usually
             sp.sympify(expression)
        return {"ok": True, "result": "42"}
    except Exception as e:
        return {"ok": False, "error": e}

def test_repl_logic():
    text = 'print("Hello world")'
    res = evaluate_safely(text)
    
    if res.get("ok"):
        print("Result:", res.get("result"))
    else:
        # Logic copied from repl_core.py fix
        err = res.get('error', '')
        if "syntax" in str(err).lower() or "invalid syntax" in str(err).lower():
            print(f"Error: Invalid syntax in '{text}'. (Only mathematical expressions are supported)")
        else:
            print(f"Error: {err}")

if __name__ == "__main__":
    print("--- Test Output ---")
    test_repl_logic()
