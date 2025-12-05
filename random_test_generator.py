#!/usr/bin/env python
"""Random test generator for Kalkulator - generates random expressions and tests them."""
import random
import sys

sys.path.insert(0, ".")

import traceback

from kalkulator_pkg.solver import solve_single_equation, solve_system
from kalkulator_pkg.worker import evaluate_safely

# Available functions and operations
FUNCTIONS = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "ln",
    "log",
    "exp",
    "sqrt",
    "Abs",
    "diff",
    "integrate",
    "Matrix",
    "det",
    "factor",
    "expand",
    "simplify",
]

CONSTANTS = ["pi", "E", "I"]

OPERATORS = ["+", "-", "*", "/", "^", "**"]

VARIABLES = ["x", "y", "z", "a", "b", "c", "t", "u", "v", "w"]


def generate_random_number():
    """Generate a random number."""
    if random.random() < 0.3:
        # Integer
        return str(random.randint(-100, 100))
    elif random.random() < 0.5:
        # Decimal
        return f"{random.uniform(-100, 100):.3f}"
    else:
        # Fraction
        num = random.randint(-50, 50)
        den = random.randint(1, 50)
        return f"{num}/{den}"


def generate_random_expression(max_depth=3, max_length=200):
    """Generate a random mathematical expression."""
    if max_depth <= 0 or max_length <= 0:
        # Base case: return a simple term
        choice = random.random()
        if choice < 0.3:
            return generate_random_number()
        elif choice < 0.5:
            return random.choice(VARIABLES)
        elif choice < 0.7:
            return random.choice(CONSTANTS)
        else:
            # Function call
            func = random.choice(FUNCTIONS)
            if func in ["diff", "integrate"]:
                # These need special handling with commas
                expr = generate_random_expression(max_depth - 1, max_length - 20)
                var = random.choice(VARIABLES)
                return f"{func}({expr}, {var})"
            else:
                arg = generate_random_expression(max_depth - 1, max_length - 20)
                return f"{func}({arg})"

    # Non-base case: build expression
    choice = random.random()

    if choice < 0.2:
        # Binary operation
        left = generate_random_expression(max_depth - 1, max_length // 2)
        op = random.choice(OPERATORS)
        right = generate_random_expression(max_depth - 1, max_length // 2)
        return f"({left}){op}({right})"
    elif choice < 0.3:
        # Unary operation (negation)
        expr = generate_random_expression(max_depth - 1, max_length - 5)
        return f"-({expr})"
    elif choice < 0.5:
        # Function call
        func = random.choice(FUNCTIONS)
        if func in ["diff", "integrate"]:
            expr = generate_random_expression(max_depth - 1, max_length - 20)
            var = random.choice(VARIABLES)
            return f"{func}({expr}, {var})"
        elif func == "Matrix":
            # Matrix needs special format
            size = random.randint(2, 3)
            rows = []
            for _ in range(size):
                row = [generate_random_number() for _ in range(size)]
                rows.append(f"[{','.join(row)}]")
            return f"Matrix([{','.join(rows)}])"
        elif func == "det":
            # Determinant needs a matrix
            size = random.randint(2, 3)
            rows = []
            for _ in range(size):
                row = [generate_random_number() for _ in range(size)]
                rows.append(f"[{','.join(row)}]")
            return f"det(Matrix([{','.join(rows)}]))"
        else:
            arg = generate_random_expression(max_depth - 1, max_length - 20)
            return f"{func}({arg})"
    elif choice < 0.7:
        # Simple term
        return random.choice(
            [
                generate_random_number(),
                random.choice(VARIABLES),
                random.choice(CONSTANTS),
            ]
        )
    else:
        # Nested expression
        expr = generate_random_expression(max_depth - 1, max_length - 10)
        return f"({expr})"


def generate_random_equation():
    """Generate a random equation."""
    left = generate_random_expression(max_depth=2, max_length=100)
    right = generate_random_expression(max_depth=2, max_length=100)
    return f"{left}={right}"


def generate_random_system():
    """Generate a random system of equations."""
    num_eqs = random.randint(2, 4)
    equations = [generate_random_equation() for _ in range(num_eqs)]
    return ", ".join(equations)


def test_expression(expr):
    """Test an expression. Returns (success, error_type, error_msg).
    success=True means no bugs found.
    """
    try:
        result = evaluate_safely(expr)
        if result.get("ok"):
            return True, None, None
        else:
            error = result.get("error", "Unknown error")
            # Parse errors are bugs
            if "parse" in error.lower():
                return False, "error", error
            # Other errors might be expected (domain errors, etc.)
            # But let's be conservative and flag them
            return False, "error", error
    except Exception as e:
        return False, "exception", str(e)


def test_equation(eq_str):
    """Test an equation. Returns (success, error_type, error_msg).
    success=True means no bugs found (even if no solution exists, that's expected).
    """
    try:
        # Check if it has "find"
        if "find" in eq_str.lower():
            import re

            find_tokens = re.findall(r"\bfind\s+(\w+)\b", eq_str, re.IGNORECASE)
            find_var = find_tokens[0] if find_tokens else None
            eq_str = re.sub(
                r",?\s*\bfind\s+\w+\b\s*,?", "", eq_str, flags=re.IGNORECASE
            ).strip()
            eq_str = eq_str.rstrip(",").strip()

        result = solve_single_equation(eq_str, None)
        if result.get("ok"):
            return True, None, None
        else:
            error = result.get("error", "Unknown error")
            # "No solution" is expected, not a bug
            if "no solution" in error.lower() or "no real solution" in error.lower():
                return True, None, None
            # Parse errors, exceptions are bugs
            if "parse" in error.lower() or "error" in error.lower():
                return False, "error", error
            # Other errors might be bugs
            return False, "error", error
    except Exception as e:
        return False, "exception", str(e)


def test_system(sys_str):
    """Test a system. Returns (success, error_type, error_msg).
    success=True means no bugs found (even if no solution exists, that's expected).
    """
    try:
        import re

        find_tokens = re.findall(r"\bfind\s+(\w+)\b", sys_str, re.IGNORECASE)
        find_token = find_tokens[0] if find_tokens else None
        raw_no_find = re.sub(
            r"\bfind\s+\w+\b", "", sys_str, flags=re.IGNORECASE
        ).strip()
        result = solve_system(raw_no_find, find_token)
        if result.get("ok"):
            return True, None, None
        else:
            error = result.get("error", "Unknown error")
            # "No solution" is expected, not a bug
            if (
                "no solution" in error.lower()
                or "no real solution" in error.lower()
                or "inconsistent" in error.lower()
            ):
                return True, None, None
            # Parse errors, exceptions are bugs
            if "parse" in error.lower() or "error" in error.lower():
                return False, "error", error
            # Other errors might be bugs
            return False, "error", error
    except Exception as e:
        return False, "exception", str(e)


def main():
    """Run random tests."""
    random.seed(42)  # For reproducibility
    total_tests = 100
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("RANDOM TEST GENERATOR FOR KALKULATOR")
    print("=" * 70)
    print(f"Running {total_tests} random tests...")
    print()

    for i in range(1, total_tests + 1):
        test_type = random.choice(["expression", "equation", "system"])

        try:
            if test_type == "expression":
                expr = generate_random_expression(
                    max_depth=random.randint(1, 3), max_length=random.randint(50, 200)
                )
                success, error_type, error = test_expression(expr)
                test_str = expr
            elif test_type == "equation":
                eq = generate_random_equation()
                success, error_type, error = test_equation(eq)
                test_str = eq
            else:  # system
                sys = generate_random_system()
                success, error_type, error = test_system(sys)
                test_str = sys

            if success:
                passed += 1
                if i % 10 == 0:
                    print(f"[{i}/{total_tests}] OK: {test_str[:60]}...")
            else:
                failed += 1
                errors.append((i, test_str, error_type, error))
                print(
                    f"[{i}/{total_tests}] BUG FOUND [{error_type}]: {test_str[:60]}..."
                )
                print(f"         Error: {error}")

        except Exception as e:
            failed += 1
            errors.append(
                (
                    i,
                    test_str if "test_str" in locals() else "unknown",
                    "exception",
                    str(e),
                )
            )
            print(f"[{i}/{total_tests}] CRASH: {e}")
            traceback.print_exc()

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {total_tests}")
    print(f"Success Rate: {100*passed/total_tests:.1f}%")

    if errors:
        print()
        print("BUGS FOUND:")
        for i, test_str, error_type, error in errors[:20]:  # Show first 20
            print(f"  [{i}] [{error_type}] {test_str[:50]}...")
            print(f"      Error: {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more bugs")
    else:
        print()
        print("✓ No bugs found! All tests passed.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
