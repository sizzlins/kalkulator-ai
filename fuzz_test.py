#!/usr/bin/env python3
"""Fuzz testing for Kalkulator - generate random inputs to find bugs."""

import random
import subprocess
import sys
import time
from typing import List, Tuple

# Test configuration
NUM_TESTS = 100
MAX_ITERATIONS = 1000
TIMEOUT = 3  # seconds per test
SYSTEM_TIMEOUT = 2  # shorter timeout for system equations (can be slow)

# Operation types to test
OPERATIONS = [
    "+", "-", "*", "/", "**", "^"
]

FUNCTIONS = [
    "sin", "cos", "tan", "asin", "acos", "atan",
    "ln", "log", "exp", "sqrt", "abs"
]

CONSTANTS = ["pi", "E", "I", "2", "3", "5", "10", "100"]

VARIABLES = ["x", "y", "z", "a", "b", "c", "t", "u", "v", "w"]

def random_float() -> str:
    """Generate a random float string."""
    if random.random() < 0.3:
        return str(random.randint(-100, 100))
    elif random.random() < 0.5:
        return f"{random.randint(-100, 100)}.{random.randint(0, 999)}"
    else:
        return f"{random.randint(-10, 10)}.{random.randint(0, 999999)}"

def random_expression(depth: int = 0, max_depth: int = 3) -> str:
    """Generate a random mathematical expression."""
    if depth >= max_depth:
        if random.random() < 0.3:
            return random.choice(VARIABLES)
        elif random.random() < 0.5:
            return random_float()
        else:
            return random.choice(CONSTANTS)
    
    op = random.choice(OPERATIONS)
    if op in ["**", "^"]:
        # Avoid deep nesting with powers
        left = random_expression(depth + 1, min(max_depth, depth + 2))
        right = random.choice(["2", "3", "x", "y"])
    else:
        left = random_expression(depth + 1, max_depth)
        right = random_expression(depth + 1, max_depth)
    
    return f"({left} {op} {right})"

def random_function_call() -> str:
    """Generate a random function call."""
    func = random.choice(FUNCTIONS)
    if random.random() < 0.5:
        arg = random_expression(max_depth=2)
    else:
        arg = random.choice(VARIABLES + [random_float()])
    return f"{func}({arg})"

def random_equation() -> str:
    """Generate a random equation."""
    left = random_expression(max_depth=2)
    right = random_expression(max_depth=2)
    return f"{left} = {right}"

def random_system() -> str:
    """Generate a random system of equations."""
    # Limit to 2 equations to avoid O(n!) complexity in solver
    num_eqs = 2
    eqs = [random_equation() for _ in range(num_eqs)]
    return ", ".join(eqs)

def random_function_definition() -> str:
    """Generate a random function definition."""
    func_name = random.choice(["f", "g", "h", "F", "G"])
    num_params = random.randint(1, 3)
    params = random.sample(VARIABLES, num_params)
    body = random_expression(max_depth=2)
    return f"{func_name}({','.join(params)})={body}"

def random_function_call_eval() -> str:
    """Generate a random function call for evaluation."""
    func_name = random.choice(["f", "g", "h"])
    num_args = random.randint(1, 3)
    args = [random_float() for _ in range(num_args)]
    return f"{func_name}({','.join(args)})"

def random_function_finding() -> str:
    """Generate a random function finding command."""
    func_name = random.choice(["f", "g", "h"])
    num_params = random.randint(1, 2)
    params = random.sample(VARIABLES, num_params)
    
    num_points = random.randint(2, 4)
    points = []
    for _ in range(num_points):
        args = [random_float() for _ in range(num_params)]
        value = random_float()
        points.append(f"{func_name}({','.join(args)})={value}")
    
    return f"{', '.join(points)}, find {func_name}({','.join(params)})"

def random_integration() -> str:
    """Generate a random integration expression."""
    expr = random_expression(max_depth=2)
    var = random.choice(VARIABLES)
    return f"integrate({expr}, {var})"

def random_differentiation() -> str:
    """Generate a random differentiation expression."""
    expr = random_expression(max_depth=2)
    var = random.choice(VARIABLES)
    return f"diff({expr}, {var})"

def random_matrix() -> str:
    """Generate a random matrix expression."""
    size = random.randint(2, 3)
    rows = []
    for _ in range(size):
        row = [random.choice(["1", "0", "2", "-1", "x"]) for _ in range(size)]
        rows.append(f"[{','.join(row)}]")
    return f"Matrix([{','.join(rows)}])"

def generate_test_input() -> Tuple[str, str]:
    """Generate a random test input."""
    test_type = random.choices(
        ["expression", "equation", "system", "function_def", "function_call",
         "function_finding", "integration", "differentiation", "matrix", "complex"],
        weights=[3, 2, 1, 2, 2, 1, 1, 1, 1, 1]
    )[0]
    
    if test_type == "expression":
        if random.random() < 0.3:
            expr = random_function_call()
        else:
            expr = random_expression()
        return (expr, "expression")
    
    elif test_type == "equation":
        return (random_equation(), "equation")
    
    elif test_type == "system":
        return (random_system(), "system")
    
    elif test_type == "function_def":
        return (random_function_definition(), "function_def")
    
    elif test_type == "function_call":
        return (random_function_call_eval(), "function_call")
    
    elif test_type == "function_finding":
        return (random_function_finding(), "function_finding")
    
    elif test_type == "integration":
        return (random_integration(), "integration")
    
    elif test_type == "differentiation":
        return (random_differentiation(), "differentiation")
    
    elif test_type == "matrix":
        return (random_matrix(), "matrix")
    
    else:  # complex
        parts = []
        if random.random() < 0.5:
            parts.append(random_function_definition())
        parts.append(random_expression())
        if random.random() < 0.3:
            parts.append(random_function_call())
        return (", ".join(parts), "complex")

def run_test(expr: str) -> Tuple[bool, str, str]:
    """Run a single test and return (success, output, error)."""
    try:
        # Use --eval mode to test
        # Use shorter timeout for system equations
        current_timeout = SYSTEM_TIMEOUT if "=" in expr and expr.count(",") >= 1 else TIMEOUT
        cmd = [sys.executable, "kalkulator.py", "--eval", expr]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=current_timeout,
            cwd="."
        )
        
        output = result.stdout.strip()
        error = result.stderr.strip()
        
        # Check for fatal errors or exceptions
        if result.returncode != 0:
            if "Fatal error" in output or "Fatal error" in error:
                return (False, output, error)
            if "Traceback" in output or "Traceback" in error:
                return (False, output, error)
        
        # Check for common error patterns that shouldn't crash
        fatal_patterns = [
            "cannot access local variable",
            "UnboundLocalError",
            "NameError",
            "AttributeError",
            "TypeError",
            "ValueError",
            "IndexError",
            "KeyError",
            "ZeroDivisionError",
            "RecursionError",
            "MemoryError",
            "TimeoutError"
        ]
        
        full_output = output + " " + error
        for pattern in fatal_patterns:
            if pattern in full_output and "Error:" not in full_output:
                # If it's a fatal error not caught by error handling
                if "Fatal error" not in full_output:
                    return (False, output, error)
        
        return (True, output, error)
        
    except subprocess.TimeoutExpired:
        return (False, "", "Timeout")
    except Exception as e:
        return (False, "", str(e))

def main():
    """Main fuzzing loop."""
    print(f"Starting fuzz testing with {NUM_TESTS} tests...")
    print("=" * 70)
    
    failures = []
    successes = 0
    
    for i in range(NUM_TESTS):
        expr, test_type = generate_test_input()
        
        # Skip very long expressions that might cause issues
        if len(expr) > 500:
            continue
        
        print(f"\n[{i+1}/{NUM_TESTS}] Testing ({test_type}): {expr}...")
        
        success, output, error = run_test(expr)
        
        if success:
            successes += 1
            print(f"  ✓ PASS")
        else:
            failures.append((expr, test_type, output, error))
            print(f"  ✗ FAIL")
            if output:
                print(f"    Output: {output[:200]}")
            if error:
                print(f"    Error: {error[:200]}")
    
    print("\n" + "=" * 70)
    print(f"Fuzz Test Results:")
    print(f"  Total tests: {NUM_TESTS}")
    print(f"  Passed: {successes}")
    print(f"  Failed: {len(failures)}")
    
    if failures:
        print("\n" + "=" * 70)
        print("FAILURES:")
        print("=" * 70)
        for i, (expr, test_type, output, error) in enumerate(failures, 1):
            print(f"\n{i}. Type: {test_type}")
            print(f"   Input: {expr}")
            if output:
                print(f"   Output: {output[:300]}")
            if error:
                print(f"   Error: {error[:300]}")
            print("-" * 70)
    
    return len(failures)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 255))

