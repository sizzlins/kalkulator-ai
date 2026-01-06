"""Input parsing and preprocessing module.

This module handles:
- Input sanitization and validation
- Expression preprocessing (symbol conversion, exponent handling, etc.)
- SymPy expression parsing with security validation
- Result formatting (superscripts, numbers, solutions)
- Balancing checks for parentheses/brackets
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import sympy as sp
from sympy import parse_expr

from .config import ALLOWED_SYMPY_NAMES
from .config import AMBIG_FRACTION_REGEX
from .config import CACHE_SIZE_PARSE
from .config import DIGIT_LETTERS_REGEX
from .config import MAX_EXPRESSION_DEPTH
from .config import MAX_EXPRESSION_NODES
from .config import MAX_INPUT_LENGTH
from .config import OUTPUT_PRECISION
from .config import PERCENT_REGEX
from .config import SQRT_UNICODE_REGEX
from .config import TRANSFORMATIONS
from .types import ValidationError

# Minimal globals for SymPy literals to prevent namespace pollution
SAFE_GLOBALS = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Pow": sp.Pow,
    "Add": sp.Add,
    "Mul": sp.Mul,
}


def superscriptify(input_str: str) -> str:
    """Convert numeric string to Unicode superscript characters.

    Args:
        input_str: Input string with digits and '-' (e.g., "123", "-5")

    Returns:
        String with superscript Unicode characters (e.g., "¹²³", "⁻⁵")
    """
    mapping = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
        "n": "ⁿ",
    }
    return "".join(mapping.get(char, char) for char in input_str)


def format_superscript(expr_str: str) -> str:
    """Replace Python power notation (**) with Unicode superscripts.

    Args:
        expr_str: Expression string (e.g., "x**2", "x**-3")

    Returns:
        String with superscripts (e.g., "x²", "x⁻³")
    """
    return re.sub(r"\*\*(\-?\d+)", lambda m: superscriptify(m.group(1)), expr_str)


def format_number(val: Any, precision: int = OUTPUT_PRECISION) -> str:
    """Format a numeric value with specified precision.

    Args:
        val: Numeric value to format
        precision: Number of significant digits (default: OUTPUT_PRECISION)

    Returns:
        Formatted string representation of the number
    """
    try:
        num = float(val)
        # Use fixed-point notation with high precision to preserve exact values
        # Format with enough decimal places to capture precision
        formatted = f"{num:.{precision}f}"
        # Remove trailing zeros and decimal point if not needed
        formatted = formatted.rstrip('0').rstrip('.')
        return formatted
    except (ValueError, TypeError, OverflowError):
        # Fallback for non-numeric or invalid values
        return str(val)


def format_solution(sol: Any, exact: bool = True) -> str:
    """Format a solution value or tuple for display.

    Args:
        sol: Solution value(s) - can be a number, tuple, or list
        exact: If True, use superscript formatting; if False, use numeric formatting

    Returns:
        Formatted string representation
    """
    if isinstance(sol, (tuple, list)):
        return "(" + ", ".join(format_solution(v, exact) for v in sol) + ")"
    return format_superscript(str(sol)) if exact else format_number(sol)


def prettify_expr(expr_str: str) -> str:
    """Convert expression string to more readable format.

    Replaces 'sqrt(' with '√' and '*' with '×' for better readability.

    Args:
        expr_str: Expression string (e.g., "sqrt(4)*2")

    Returns:
        Prettified string (e.g., "√(4)×2")
    """
    result = re.sub(r"sqrt\(([^)]+)\)", r"√(\1)", expr_str)
    result = result.replace("*", "×")
    return result


def is_balanced(input_str: str) -> tuple[bool, int | None]:
    """Check if parentheses/brackets are balanced. Returns (is_balanced, error_position)."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[tuple[str, int]] = []  # (char, position)
    for i, char in enumerate(input_str):
        if char in pairs:
            stack.append((char, i))
        elif char in pairs.values():
            if not stack:
                return False, i
            opening, pos = stack.pop()
            if pairs[opening] != char:
                return False, i
    if stack:
        return False, stack[0][1]  # Return position of first unmatched
    return True, None


# Command names that should not appear in mathematical expressions
REPL_COMMANDS = {
    "showcache",
    "clearcache",
    "savecache",
    "loadcache",
    "timing",
    "cachehits",
    "showcachehits",
    "help",
    "quit",
    "exit",
    "eval",  # For --eval command
    "savefunction",
    "savefunctions",
    "loadfunction",
    "loadfunctions",
    "clearfunction",
    "clearfunctions",
    "clearsavefunction",
    "clearsavefunctions",
    "showfunction",
    "showfunctions",
    "list",
    "export",
    "evolve",
    "benchmark",
}

# Basic denylist to avoid dangerous tokens before SymPy parsing
FORBIDDEN_TOKENS = (
    "__",
    "import",
    "lambda",
    "eval",
    "exec",
    "open",
    "os.",
    "sys.",
    "subprocess",
    "builtins",
    "getattr",
    "setattr",
    "delattr",
    "compile",
    "globals",
    "locals",
    "__class__",
    "__mro__",
    "__subclasses__",
    "memoryview",
    "bytes",
    "bytearray",
    "__import__",
)


def _validate_expression_tree(
    expr: Any,
    depth: int = 0,
    node_count: list[int] = None,
    allow_none: bool = False,
    allowed_functions: set[str] | None = None,
) -> None:
    """Validate expression tree structure - reject dangerous nodes.

    Args:
        expr: Expression to validate
        depth: Current depth in the tree
        node_count: List to track total node count (modified in place)
        allow_none: If True, allow None as a valid result
        allowed_functions: Optional set of extra function names to allow (e.g., 'f', 'g' for function finding)
    """
    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if node_count[0] > MAX_EXPRESSION_NODES:
        raise ValidationError(
            f"Expression too complex (>{MAX_EXPRESSION_NODES} nodes)", "TOO_COMPLEX"
        )
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValidationError(
            f"Expression too deeply nested (>{MAX_EXPRESSION_DEPTH} levels)", "TOO_DEEP"
        )

    # Allow None at top level (depth 0) if explicitly allowed
    # This handles cases like print() which execute successfully but return None
    if expr is None and allow_none and depth == 0:
        return

    # Allow safe types - Numbers (includes Integer, Rational, Float, Complex)
    if isinstance(expr, (sp.Symbol, sp.Number)):
        return
    if isinstance(expr, sp.Function):
        # Only allow whitelisted functions
        # Try multiple methods to get function name
        func_name = None
        if hasattr(expr.func, "__name__"):
            func_name = expr.func.__name__
        elif hasattr(expr.func, "name"):
            func_name = expr.func.name
        else:
            # Fallback: extract from string representation
            func_str = str(expr.func)
            if "." in func_str:
                func_name = func_str.split(".")[-1].split("'")[0]
            else:
                func_name = func_str.split("'")[1] if "'" in func_str else func_str

        # Check against basic allowed list first
        is_allowed = func_name and func_name in ALLOWED_SYMPY_NAMES

        # If not in allowed names, check optional whitelist
        if not is_allowed and allowed_functions:
            is_allowed = func_name in allowed_functions

        if func_name and not is_allowed:
            # Audit log blocked function
            try:
                from .logging_config import get_logger

                logger = get_logger("parser")
                logger.warning(
                    "Blocked forbidden function",
                    extra={"forbidden_function": func_name},
                )
            except ImportError:
                pass
            raise ValidationError(
                f"Function '{func_name}' not allowed", "FORBIDDEN_FUNCTION"
            )
        # Recurse into args
        for arg in expr.args:
            _validate_expression_tree(arg, depth + 1, node_count)
        return
    # Allow safe arithmetic operations
    if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
        for arg in expr.args:
            _validate_expression_tree(arg, depth + 1, node_count)
        return
    # Handle special SymPy singleton objects (they're still Numbers)
    # Check if it's a well-known singleton value
    try:
        if expr in (
            sp.S.One,
            sp.S.Zero,
            sp.S.NegativeOne,
            sp.S.Half,
            sp.S.NaN,
            sp.oo,
            -sp.oo,
        ):
            return
    except (ValueError, TypeError, AttributeError):
        # Expected for some singleton checks
        pass
    if isinstance(expr, sp.Matrix):
        for row in expr.tolist():
            for elem in row:
                _validate_expression_tree(
                    elem, depth + 1, node_count, allowed_functions=allowed_functions
                )
        return
    # Check for dangerous types explicitly
    expr_type = type(expr).__name__
    # Reject Attribute access (could expose internals)
    if expr_type == "Attribute":
        raise ValidationError(
            f"Dangerous expression type '{expr_type}' not allowed", "FORBIDDEN_TYPE"
        )

        return

    # Allow lists and tuples (for data sets)
    if expr_type in ("list", "tuple") or isinstance(expr, (list, tuple)):
        # Validate all items in the container
        for item in expr:
            _validate_expression_tree(
                item, depth + 1, node_count, allowed_functions=allowed_functions
            )
        return

    # Allow primitive types (recursion hits these for contents of lists)
    if expr_type in ("int", "float", "str", "complex", "bool") or isinstance(
        expr, (int, float, str, complex, bool)
    ):
        return

    # Allow other SymPy Basic types (they're generally safe)
    if isinstance(expr, sp.Basic):
        # For expressions with args, validate children
        if hasattr(expr, "args") and expr.args:
            for arg in expr.args:
                _validate_expression_tree(
                    arg, depth + 1, node_count, allowed_functions=allowed_functions
                )
        # For relational operators, validate both sides
        if hasattr(expr, "lhs") and hasattr(expr, "rhs"):
            _validate_expression_tree(
                expr.lhs, depth + 1, node_count, allowed_functions=allowed_functions
            )
            _validate_expression_tree(
                expr.rhs, depth + 1, node_count, allowed_functions=allowed_functions
            )
        return

    # Reject anything that's not a SymPy Basic type
    raise ValidationError(
        f"Expression type '{expr_type}' not allowed", "FORBIDDEN_TYPE"
    )


def _protect_function_commas(expr: str) -> tuple[str, dict[str, str]]:
    """Protect commas inside function calls from being parsed as tuples.

    For expressions like integrate(1/x, x) or diff(sin(x), x), we need to
    protect the commas inside function calls so they don't get parsed as tuple creation.

    Args:
        expr: Expression string

    Returns:
        Tuple of (protected_expression, replacements_dict) where replacements_dict
        maps placeholder strings back to original commas
    """
    import re

    replacements = {}
    placeholder_counter = [0]

    def create_placeholder():
        placeholder_counter[0] += 1
        return f"__COMMA_PLACEHOLDER_{placeholder_counter[0]}__"

    # Pattern to match function calls: function_name(...)
    # We need to find function calls and protect commas inside them
    # But be careful - we don't want to protect commas in nested structures incorrectly

    # Match function calls with their arguments
    # Pattern: function_name followed by parentheses with content
    func_pattern = re.compile(r"(\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)")

    def protect_func_args(match):
        func_name = match.group(1)
        args_content = match.group(2)

        # Only protect if this is a known function that takes multiple arguments
        multi_arg_funcs = {"integrate", "diff", "limit", "sum", "product"}
        if func_name in multi_arg_funcs:
            # Replace commas in arguments with placeholders
            protected = args_content
            depth = 0
            result = []
            i = 0
            while i < len(protected):
                char = protected[i]
                if char == "(":
                    depth += 1
                    result.append(char)
                elif char == ")":
                    depth -= 1
                    result.append(char)
                elif char == "," and depth == 0:
                    # This comma separates arguments at the function call level
                    placeholder = create_placeholder()
                    replacements[placeholder] = ","
                    result.append(placeholder)
                else:
                    result.append(char)
                i += 1
            return f"{func_name}({''.join(result)})"
        return match.group(0)

    protected_expr = func_pattern.sub(protect_func_args, expr)
    return protected_expr, replacements


def _restore_function_commas(expr: str, replacements: dict[str, str]) -> str:
    """Restore protected commas in function calls.

    Args:
        expr: Expression with placeholders
        replacements: Dictionary mapping placeholders to commas

    Returns:
        Expression with placeholders replaced by commas
    """
    for placeholder, comma in replacements.items():
        expr = expr.replace(placeholder, comma)
    return expr



def _strip_comments(text: str) -> str:
    """Strip comments starting with #, respecting quotes.

    Args:
        text: Input string

    Returns:
        String with comments removed
    """
    out = []
    in_quote = False
    quote_char = None
    i = 0
    while i < len(text):
        char = text[i]
        if in_quote:
            if char == quote_char:
                # Check for escape
                if i > 0 and text[i - 1] == "\\":
                    pass
                else:
                    in_quote = False
            out.append(char)
        else:
            if char == '"' or char == "'":
                in_quote = True
                quote_char = char
                out.append(char)
            elif char == "#":
                break  # Comment start
            else:
                out.append(char)
        i += 1
    return "".join(out)


def preprocess(
    input_str: str,
    skip_exponent_conversion: bool = False,
    allowed_functions: frozenset[str] | None = None,
) -> str:
    """Preprocess input string for parsing.

    Applies transformations:
    - Strips comments (# ...)
    - Validates input length and forbidden tokens
    - Standardizes mathematical symbols (unicode variants to ASCII)
    - Converts exponents (^ to **, superscripts to **)
    - Handles percentages (50% -> (50/100))
    - Converts Unicode square root (√) to sqrt(
    - Inserts implicit multiplication (2x -> 2*x)
    - Protects commas in function calls (integrate, diff, etc.)
    - Validates balanced parentheses/brackets

    Args:
        input_str: Raw input string from user
        skip_exponent_conversion: If True, skips ^ and superscript conversion

    Returns:
        Preprocessed and sanitized string ready for SymPy parsing

    Raises:
        ValidationError: If input is too long, contains forbidden tokens,
                        or has unbalanced parentheses/brackets
    """
    if not input_str:
        raise ValidationError("Input cannot be empty", "EMPTY_INPUT")

    # Strip comments first
    input_str = _strip_comments(input_str)
    
    # Input size check
    input_str = input_str.strip() if input_str else ""

    # Protect commas in multi-argument function calls (integrate, diff, etc.)
    # This MUST happen BEFORE implicit multiplication to prevent 1/x, x from becoming (1)/(x, x)
    # We'll temporarily replace commas in function calls with a special marker
    import re

    processed_str = input_str
    protected_funcs = []
    target_funcs = {"integrate", "diff", "limit", "sum", "product", "min", "max"}

    # Scan for function calls manually to handle nested parentheses correctly
    # Regex like `diff\(([^)]+)\)` fails on `diff(f(x), x)` because it stops at the first `)`
    i = 0
    result_parts = []
    last_pos = 0

    while i < len(processed_str):
        # Check for function names
        found_func = None
        for func in target_funcs:
            if processed_str.startswith(func, i):
                # Check boundaries (must be whole word)
                end_func = i + len(func)
                if (
                    end_func == len(processed_str)
                    or not processed_str[end_func].isalnum()
                ) and (i == 0 or not processed_str[i - 1].isalnum()):
                    # Check if followed by (
                    j = end_func
                    while j < len(processed_str) and processed_str[j].isspace():
                        j += 1
                    if j < len(processed_str) and processed_str[j] == "(":
                        found_func = (func, end_func, j)
                        break

        if found_func:
            func_name, end_func_pos, open_paren_pos = found_func

            # Find matching closing parenthesis
            depth = 1
            j = open_paren_pos + 1
            args_start = j

            while j < len(processed_str) and depth > 0:
                if processed_str[j] == "(":
                    depth += 1
                elif processed_str[j] == ")":
                    depth -= 1
                j += 1

            if depth == 0:
                # Found valid function call
                args_end = j - 1
                args = processed_str[args_start:args_end]

                # Check if args contain a comma (at the top level)
                arg_depth = 0
                has_top_comma = False
                for char in args:
                    if char == "(":
                        arg_depth += 1
                    elif char == ")":
                        arg_depth -= 1
                    elif char == "," and arg_depth == 0:
                        has_top_comma = True
                        break

                if has_top_comma:
                    # Protect commas
                    marker = f" __COMMA_SEP_{len(protected_funcs)}__ "
                    protected_args = ""
                    arg_depth = 0
                    for char in args:
                        if char == "(":
                            arg_depth += 1
                            protected_args += char
                        elif char == ")":
                            arg_depth -= 1
                            protected_args += char
                        elif char == "," and arg_depth == 0:
                            protected_args += marker
                        else:
                            protected_args += char

                    protected_funcs.append((func_name, args, marker))

                    # Add everything before this call
                    result_parts.append(processed_str[last_pos:i])
                    # Add the protected call
                    result_parts.append(f"{func_name}({protected_args})")

                    last_pos = j
                    i = j
                    continue

            # If depth != 0 (unbalanced) or no top comma, just continue scanning
            i = open_paren_pos + 1
        else:
            i += 1

    result_parts.append(processed_str[last_pos:])
    processed_str = "".join(result_parts)

    if not input_str:
        raise ValidationError("Input cannot be empty", "EMPTY_INPUT")
    if len(input_str) > MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input too long (>{MAX_INPUT_LENGTH} characters)", "TOO_LONG"
        )

    # Check for unterminated quotes (common syntax error)
    if input_str.count('"') % 2 != 0:
        raise ValidationError(
            "Unmatched double quotes detected. Check that all opening quotes have matching closing quotes.",
            "UNMATCHED_QUOTES",
        )
    if input_str.count("'") % 2 != 0:
        raise ValidationError(
            "Unmatched single quotes detected. Check that all opening quotes have matching closing quotes.",
            "UNMATCHED_QUOTES",
        )

    # Check for REPL command names appearing in expressions (common paste error)
    input_lower = input_str.lower()
    found_commands = []
    for cmd in REPL_COMMANDS:
        # Check if command appears as a whole word (not part of another word)
        # Use word boundaries or check that it's not part of a valid identifier

        pattern = r"\b" + re.escape(cmd) + r"\b"
        if re.search(pattern, input_lower, re.IGNORECASE):
            # Make sure it's not part of a longer valid identifier or function
            # Check context: should not be preceded/followed by alphanumeric or underscore
            matches = list(re.finditer(pattern, input_lower, re.IGNORECASE))
            for match in matches:
                start, end = match.span()
                # Check if it's actually a standalone word
                if (start == 0 or not input_str[start - 1].isalnum()) and (
                    end == len(input_str)
                    or not (input_str[end].isalnum() or input_str[end] == "_")
                ):
                    found_commands.append(cmd)
                    break
    if found_commands:
        commands_str = ", ".join(sorted(set(found_commands)))
        raise ValidationError(
            f"Command name(s) detected in expression: {commands_str}. "
            "Commands cannot be used in mathematical expressions. "
            "Did you paste multiple commands? Please enter them separately, one per line.",
            "COMMAND_IN_EXPRESSION",
        )

    lowered = input_str.strip().lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in lowered:
            # Audit log blocked input
            try:
                from .logging_config import get_logger

                logger = get_logger("parser")
                logger.warning(
                    "Blocked input containing forbidden token",
                    extra={
                        "forbidden_token": tok,
                        "input_length": len(input_str),
                        "input_preview": (
                            input_str[:100]
                            if len(input_str) <= 100
                            else input_str[:100] + "..."
                        ),
                    },
                )
            except ImportError:
                pass
            raise ValidationError(
                f"Input contains forbidden token: {tok}", "FORBIDDEN_TOKEN"
            )

    # Note: processed_str may have been modified by function comma protection above
    if "processed_str" not in locals():
        processed_str = input_str.strip()
    processed_str = processed_str.replace("−", "-").replace("–", "-")
    processed_str = processed_str.replace("Δ", "Delta")
    processed_str = processed_str.replace("π", "pi")  # Convert Greek π to pi

    # Replace i with I (imaginary unit) and e with E (Euler's number) if they are standalone words
    # This ensures they are treated as constants, not variables
    processed_str = re.sub(r"\bi\b", "I", processed_str)
    processed_str = re.sub(r"\be\b", "E", processed_str)

    processed_str = processed_str.replace(":", "/")
    processed_str = processed_str.replace("×", "*")
    processed_str = processed_str.replace("=>", ">=")
    processed_str = processed_str.replace("=<", "<=")

    # Handle 'mod' operator (e.g. "x mod y" -> "Mod(x, y)")
    # Must be done BEFORE implicit multiplication to avoid "x*mod*y"
    # Negative lookbehind to ensure we don't match 'mod' in 'model' or 'mode' if those become vars
    # But 'mod' is a common operator. We'll require whitespace around it or clear boundaries.
    # Pattern: something mod something
    # We use a robust regex that captures the operands.
    # Note: This is a simple infix to prefix conversion. Nested mods might need more care, but this covers standard usage.
    # We use re.sub with a function to handle potential nesting if we processed right-to-left, but simple replacement works for single level.
    # To be safe against "model", we use \bmod\b

    # We need to capture the operands. This is tricky with simple regex because operands can be complex expressions.
    # Simplified approach: Replace " mod " with ", " and wrap in Mod() is hard.
    # Better approach: Replace "\bmod\b" with "%" which SymPy might handle? No, SymPy uses % for Mod on integers but Mod(a,b) is safer for symbolic.
    # Actually, let's just replace " mod " with "%". preprocess will later handle "%" ?
    # Wait, existing code has PERCENT_REGEX.sub(r"(\1/100)", processed_str).
    # Does SymPy parse "a % b" as Mod(a, b)? Yes, Python syntax.
    # Let's test this hypothesis with a script? No, I'll trust standard Python/SymPy behavior for %.
    # So "x mod y" -> "x % y".
    # BUT we must NOT replace "mod(" (function call) with "%(" (invalid syntax).
    # Regex: mod word boundary, NOT followed by optional space and (
    processed_str = re.sub(r"\bmod\b(?!\s*\()", "%", processed_str, flags=re.IGNORECASE)

    if not skip_exponent_conversion:
        processed_str = processed_str.replace("^", "**")
        from_superscript_map = {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁻": "-",
            "ⁿ": "n",
        }
        pattern = re.compile(f"([{''.join(from_superscript_map.keys())}]+)")

        def from_superscript_converter(m):
            normal_str = "".join([from_superscript_map[char] for char in m.group(1)])
            return f"**{normal_str}"

        processed_str = pattern.sub(from_superscript_converter, processed_str)

    processed_str = PERCENT_REGEX.sub(r"(\1/100)", processed_str)
    processed_str = SQRT_UNICODE_REGEX.sub("sqrt(", processed_str)

    # AMBIG_FRACTION_REGEX.sub(r"((\1)/(\2))", processed_str)
    # Use a callback to prevent matching across protected commas
    def normalize_fraction(match):
        full_match = match.group(0)
        # If the match spans across a protected comma, do NOT transform it
        if "__COMMA_SEP_" in full_match:
            return full_match
        g1 = match.group(1)
        g2 = match.group(2)
        return f"(({g1})/({g2}))"

    processed_str = AMBIG_FRACTION_REGEX.sub(normalize_fraction, processed_str)

    # Detect and handle hexadecimal numbers BEFORE implicit multiplication
    # This prevents "123edc09f2" from becoming "123*e*d*c*0*9*f*2"
    # SymPy doesn't handle hex numbers well, so we convert to decimal before parsing

    # Also handle hex numbers with 0x prefix first - convert to decimal for SymPy compatibility
    hex_with_prefix_pattern = re.compile(r"\b0x([0-9a-fA-F]+)\b")

    def convert_0x_hex(match):
        """Convert 0x hex number to decimal."""
        hex_digits = match.group(1)
        try:
            decimal_val = int(hex_digits, 16)
            return str(decimal_val)
        except ValueError:
            return match.group(0)

    processed_str = hex_with_prefix_pattern.sub(convert_0x_hex, processed_str)

    # Detect hex numbers in assignment contexts (e.g., "var = 123edc09f2")
    # Also detect standalone hex numbers (e.g., just "123edc09f2")
    # This must happen BEFORE implicit multiplication
    # Pattern: hex digits (4+ chars) with at least one letter (a-f), in numeric context
    hex_standalone_pattern = re.compile(r"\b([0-9a-fA-F]{4,})\b")

    def convert_hex_if_valid(match):
        """Convert hex number to decimal if it looks like one."""
        hex_digits = match.group(1)
        start_pos = match.start()
        end_pos = match.end()

        # Check for preceding dot (part of float)
        if start_pos > 0 and processed_str[start_pos - 1] == ".":
            return match.group(0)

        # Only convert if it has letters (a-f) - this distinguishes hex from decimal
        if (
            len(hex_digits) >= 4
            and any(c in "abcdefABCDEF" for c in hex_digits)
            and all(c in "0123456789abcdefABCDEF" for c in hex_digits)
        ):
            # Check context: must be in a numeric context (not part of a variable name)
            # Check if followed by alphanumeric (would be part of variable) or preceded by letter/underscore
            if end_pos < len(processed_str) and (
                processed_str[end_pos].isalnum() or processed_str[end_pos] == "_"
            ):
                # Part of a longer identifier, don't convert
                return match.group(0)
            if start_pos > 0 and (
                processed_str[start_pos - 1].isalnum()
                or processed_str[start_pos - 1] == "_"
            ):
                # Preceded by letter/underscore, might be part of variable
                return match.group(0)
            try:
                # Convert hex to decimal
                decimal_val = int(hex_digits, 16)
                return str(decimal_val)
            except ValueError:
                return match.group(0)
        return match.group(0)

    # Apply hex conversion to standalone hex numbers
    processed_str = hex_standalone_pattern.sub(convert_hex_if_valid, processed_str)

    # Also handle hex in assignment contexts explicitly (for clarity)
    hex_context_pattern = re.compile(r"=\s*([0-9a-fA-F]{4,})(?=[\s\+\-\*/\)\s,;]|$)")

    def convert_hex_in_assignment(match):
        """Convert hex number after = sign to decimal."""
        hex_digits = match.group(1)
        # Only convert if it looks like a hex number (has letters a-f)
        if (
            len(hex_digits) >= 4
            and any(c in "abcdefABCDEF" for c in hex_digits)
            and all(c in "0123456789abcdefABCDEF" for c in hex_digits)
        ):
            try:
                # Convert hex to decimal
                decimal_val = int(hex_digits, 16)
                return f"= {decimal_val}"
            except ValueError:
                return match.group(0)
        return match.group(0)

    # Apply hex conversion in assignment contexts (shouldn't be needed if standalone pattern works, but for safety)
    processed_str = hex_context_pattern.sub(convert_hex_in_assignment, processed_str)

    # Protect "find" from implicit multiplication conversion
    # "find" is a keyword for function finding, not a mathematical expression
    
    def implicit_mult_replacement(match):
        """Handle implicit multiplication, respecting scientific notation."""
        digit = match.group(1)
        char = match.group(2)
        
        # If the character is 'e' or 'E', check if it looks like scientific notation
        if char.lower() == 'e':
            # Look ahead in the string to see if it's followed by digits or +,- then digits
            # Note: Regex sub only gives us the match, not the full context easily without passing string
            # But the match object has .string and .end()
            full_str = match.string
            end_idx = match.end()
            
            # Check characters immediately following 'e'
            if end_idx < len(full_str):
                next_char = full_str[end_idx]
                if next_char.isdigit() or next_char in ('+', '-'):
                    # If +, -, verify subsequent digit
                    if next_char in ('+', '-'):
                        if end_idx + 1 < len(full_str) and full_str[end_idx + 1].isdigit():
                            return match.group(0) # Keep as is (scientific notation)
                    else:
                        return match.group(0) # Keep as is (e followed by digit)

        return f"{digit}*{char}"

    if "find" in processed_str.lower():
        # Temporarily replace "find" with a placeholder
        processed_str = re.sub(
            r"\bfind\b", "___FIND_KEYWORD___", processed_str, flags=re.IGNORECASE
        )
        # Apply implicit multiplication with smart handler
        processed_str = DIGIT_LETTERS_REGEX.sub(implicit_mult_replacement, processed_str)
        # Restore "find"
        processed_str = processed_str.replace("___FIND_KEYWORD___", "find")
    else:
        processed_str = DIGIT_LETTERS_REGEX.sub(implicit_mult_replacement, processed_str)

    processed_str = re.sub(r"\s+", " ", processed_str).strip()

    # Apply sub-expression caching: replace cached sub-expressions with their values
    # This speeds up expressions like "(2+2)/2" by using cached "2+2" -> "4"
    # Example: If "2+2" is cached as "4", then "(2+2)/2" becomes "4/2" before parsing
    try:
        from .cache_manager import get_cached_subexpr

        # Strategy: Find parenthesized sub-expressions and check cache
        # Process from innermost to outermost to handle nested expressions
        paren_pattern = re.compile(r"\(([^()]+)\)")
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            matches = list(paren_pattern.finditer(processed_str))
            if not matches:
                break  # No more parentheses

            changed = False
            # Process matches from right to left to avoid index shifting issues
            for match in reversed(matches):
                subexpr = match.group(
                    1
                )  # Content inside parentheses (without the parentheses)

                # Don't replace symbolic constants with their cached numeric values
                # This preserves exact results for sin(pi), log(E), etc.
                if subexpr.strip() in ("pi", "E", "I", "zoo", "oo", "-oo"):
                    continue

                # Try to get cached value for this sub-expression
                # Note: get_cached_subexpr will track cache hits automatically
                cached_value = get_cached_subexpr(subexpr)
                if cached_value is not None and cached_value:
                    # Safety check: only replace if cached value is numeric
                    # Avoid replacing if cached value contains variables, operators, or parentheses
                    unsafe_chars = [
                        "x",
                        "y",
                        "z",
                        "X",
                        "Y",
                        "Z",
                        "a",
                        "b",
                        "c",
                        "(",
                        ")",
                        "*",
                        "/",
                        "+",
                        "-",
                        "=",
                    ]
                    if not any(c in cached_value for c in unsafe_chars):
                        # Replace the parenthesized sub-expression with its cached value
                        # Cache hit has already been tracked by get_cached_subexpr above
                        before = processed_str[: match.start()]
                        after = processed_str[match.end() :]
                        processed_str = before + "(" + cached_value + ")" + after
                        changed = True
                        break  # Restart scanning after replacement

            if not changed:
                break  # No more replacements possible
            iteration += 1
    except (ImportError, AttributeError, ValueError, TypeError):
        # If cache manager not available or error occurs, continue without sub-expression caching
        pass

    balanced, error_pos = is_balanced(processed_str)
    if not balanced:
        hint = ""
        if error_pos is not None:
            # Show context around error
            start = max(0, error_pos - 10)
            end = min(len(processed_str), error_pos + 10)
            context = processed_str[start:end]
            pointer = " " * (error_pos - start) + "^"
            hint = f" at position {error_pos}: ...{context}...\n{pointer}"
        raise ValidationError(
            f"Mismatched or unbalanced parentheses/brackets{hint}. Check parentheses around position {error_pos or 'unknown'}.",
            "UNBALANCED_PARENS",
        )

    # Expand function calls if any functions are defined
    # This happens after preprocessing but before parsing
    try:
        processed_str = expand_function_calls(processed_str)
    except ValidationError:
        # Re-raise ValidationError (especially WRONG_ARGUMENT_COUNT) so it can be displayed to user
        raise
    except Exception:
        # If function expansion fails for other reasons, continue with original string
        pass

    # Phase 4 (2025-12-10): Prevent undefined functions from becoming implicit multiplication (e.g. f(1) -> f*1)
    # After expand_function_calls, all defined user functions are replaced by their bodies.
    # So any remaining Name(...) pattern where Name is not an allowed SymPy function is likely an error.
    # Exception: We allow 'pi(x)' or 'e(x)' which parses as implicit multiplication pi*x, e*x.
    # But we strictly block unknown names.
    call_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
    for match in call_pattern.finditer(processed_str):
        name = match.group(1)
        # Check if matched name is a known SymPy function/symbol or whitelisted
        if (
            name not in ALLOWED_SYMPY_NAMES
            and (allowed_functions is None or name not in allowed_functions)
            and not name.startswith("__COMMA_SEP_")
        ):
            # Only convert to implicit multiplication if:
            # - Name is more than 1 character (e.g., "xy" -> "xy*(...)")
            # - This preserves single-letter function calls like f(x) for diff(f(x), x)
            if len(name) > 1:
                import sys

                print(
                    f"Note: Converting '{name}(...)' to '{name}*(...)' (implicit multiplication)",
                    file=sys.stderr,
                )

                # Replace name(...) with name*(...) in processed_str
                pattern_to_replace = re.compile(rf"\b{re.escape(name)}\s*\(")
                processed_str = pattern_to_replace.sub(
                    f"{name}*(", processed_str, count=1
                )

    return processed_str


@lru_cache(maxsize=CACHE_SIZE_PARSE)
def parse_preprocessed(
    expr_str: str, allowed_functions: frozenset[str] | None = None
) -> Any:
    """Cached wrapper for _parse_preprocessed_impl."""
    return _parse_preprocessed_impl(expr_str, allowed_functions, None)


def _parse_preprocessed_impl(
    expr_str: str,
    allowed_functions: frozenset[str] | None = None,
    local_dict: dict | None = None,
) -> Any:
    """Parse and validate a preprocessed expression string.

    Internal implementation that supports local_dict (which is not hashable for LRU cache).
    """
    # Handle function calls with multiple arguments that use commas
    # SymPy's parse_expr interprets commas as tuple creation
    # For integrate(expr, var) and diff(expr, var), we need to parse them specially
    # Solution: detect the pattern, parse the parts separately, then construct the call

    import re

    # Pattern to match: integrate(..., var) or diff(..., var)
    # Check for protected function call markers (from preprocessing)
    # Pattern: __COMMA_SEP_N__ where N is a number (with spaces around it)
    marker_pattern = re.compile(r"\s*__COMMA_SEP_(\d+)__\s*")

    # First, restore any markers to commas (this must happen before pattern matching)
    expr_str_restored = expr_str
    if marker_pattern.search(expr_str_restored):
        expr_str_restored = marker_pattern.sub(",", expr_str_restored)

    # Pattern to match function calls
    func_pattern = re.compile(r"^\s*(\w+)\s*\((.+)\)\s*$")

    # Check if the entire expression is a function call with commas
    # First check for markers in the original string to know where to split
    if marker_pattern.search(expr_str):
        # We have markers - restore them carefully
        # Pattern: integrate(expr __COMMA_SEP_N__ var) -> split at marker
        marker_match = marker_pattern.search(expr_str)
        if marker_match:
            # Find the function call that contains this marker
            # We need to find the function name and its opening paren
            marker_pos = marker_match.start()

            # Find the function name by looking backwards from the marker
            # Look for ANY protected function before the marker
            # This must match the list in _protect_function_commas inside preprocess
            # UPDATE: We only apply this manual splitting for integrate/diff which might have special parsing needs
            # min/max/sum/product/limit generated complex args that our manual splitter breaks (e.g. stripping parens).
            # They work fine with standard parsing via expr_str_restored.
            protected_funcs_list = ["integrate", "diff"]

            func_name_match = None
            for func_name_candidate in protected_funcs_list:
                # Find the last occurrence of the function name before the marker
                func_pos = expr_str.rfind(func_name_candidate, 0, marker_pos)
                if func_pos >= 0:
                    # Check if it's followed by an opening paren
                    after_func = expr_str[
                        func_pos + len(func_name_candidate) :
                    ].lstrip()
                    if after_func.startswith("("):
                        func_name_match = (
                            func_name_candidate,
                            func_pos,
                            func_pos + len(func_name_candidate) + after_func.index("("),
                        )
                        break

            if func_name_match:
                func_name, func_start, open_paren_pos = func_name_match

                # Find the matching closing paren for the function call first
                # This tells us the full extent of the function arguments
                depth = 1
                close_pos = open_paren_pos + 1
                while close_pos < len(expr_str) and depth > 0:
                    if expr_str[close_pos] == "(":
                        depth += 1
                    elif expr_str[close_pos] == ")":
                        depth -= 1
                    close_pos += 1

                # Now extract the arguments: everything between open_paren_pos+1 and close_pos-1
                args_str = expr_str[open_paren_pos + 1 : close_pos - 1]

                # Split at the marker position within the args
                marker_in_args_start = marker_match.start() - (open_paren_pos + 1)
                marker_in_args_end = marker_match.end() - (open_paren_pos + 1)

                # Replace the marker with a comma in the args string to reconstruct the original
                args_str_restored = (
                    args_str[:marker_in_args_start]
                    + ","
                    + args_str[marker_in_args_end:]
                )

                # Now split by the comma we just inserted
                # But we need to be careful - the comma might be inside parentheses
                # So we'll split properly by counting parentheses
                parts = []
                current = ""
                depth = 0
                for char in args_str_restored:
                    if char == "(":
                        depth += 1
                        current += char
                    elif char == ")":
                        depth -= 1
                        current += char
                    elif char == "," and depth == 0:
                        parts.append(current.strip())
                        current = ""
                    else:
                        current += char
                if current:
                    parts.append(current.strip())

                if len(parts) == 2:
                    expr_part = parts[0]
                    var_part = parts[1]

                    # Check if expr_part is incomplete (e.g., ends with '/' or incomplete paren)
                    # This happens when the marker was inside a division like (1)/(x __MARKER__ x)
                    # Pattern: (num)/(denom __MARKER__ var) -> expr = (num)/(denom), var = var
                    if expr_part.count("(") > expr_part.count(")"):
                        # The marker split a division expression - unbalanced parentheses
                        # Check if expr_part matches pattern (num)/(denom (where denom is incomplete)
                        div_match = re.match(r"^\((.+)\)/\((.+)$", expr_part)
                        if div_match:
                            num = div_match.group(1)
                            denom = div_match.group(2)
                            # var_part should be the rest of denom + ')'
                            # Check if var_part starts with denom or just 'x'
                            if (
                                var_part.startswith(denom)
                                or var_part.startswith("x")
                                or denom == "x"
                            ):
                                # Complete the division: (num)/(denom)
                                expr_part = f"({num})/({denom})"
                                # Extract the variable (should be just 'x')
                                # Remove the closing paren and any leftover denom
                                var_part = var_part.lstrip(denom).lstrip(")").strip()
                                if not var_part or var_part == ")":
                                    var_part = (
                                        "x"  # Default to x if we can't extract it
                                    )
                        elif "/" in expr_part and expr_part.count(
                            "("
                        ) > expr_part.count(")"):
                            # Generic fix: if we have unbalanced parens and a division, try to complete it
                            # This is a fallback for cases we haven't specifically handled
                            if expr_part.endswith("x") or expr_part.endswith("(x"):
                                # Likely pattern: (something)/(x -> complete to (something)/(x)
                                expr_part = expr_part + ")"
                                var_part = (
                                    var_part.lstrip("x").lstrip(")").strip() or "x"
                                )
                else:
                    # Fallback: use the marker-based split
                    expr_part = args_str[:marker_in_args_start].strip()
                    var_part = args_str[marker_in_args_end:].strip()

                    # If expr_part is incomplete (ends with / or incomplete paren), fix it
                    if expr_part.endswith("/") or expr_part.endswith("/(x"):
                        # The original was likely 1/x, so we need to complete (1)/(x to (1)/(x)
                        # But we don't know if it should be (1)/(x) or something else
                        # Try completing with the closing paren
                        if expr_part.count("(") > expr_part.count(")"):
                            expr_part = expr_part + ")"

                # Clean up var_part before parsing - remove any trailing parens or invalid characters
                var_part = var_part.rstrip(")").strip()
                if not var_part or var_part == ")" or var_part == "(":
                    var_part = "x"  # Default to x if invalid

                # Parse both parts
                try:
                    # Debug: print what we're trying to parse (commented out for production)
                    # print(f"DEBUG: Parsing expr_part={expr_part!r}, var_part={var_part!r}")

                    expr_parsed = parse_expr(
                        expr_part,
                        local_dict=ALLOWED_SYMPY_NAMES,
                        global_dict=SAFE_GLOBALS,
                        transformations=TRANSFORMATIONS,
                        evaluate=False,
                    )
                    var_parsed = parse_expr(
                        var_part,
                        local_dict=ALLOWED_SYMPY_NAMES,
                        global_dict=SAFE_GLOBALS,
                        transformations=TRANSFORMATIONS,
                        evaluate=False,
                    )

                    # Get the function
                    func = ALLOWED_SYMPY_NAMES.get(func_name)
                    if func:
                        # Call the function directly
                        result = func(expr_parsed, var_parsed)
                        # Validate the result
                        _validate_expression_tree(result)
                        return result
                except (ValueError, TypeError, SyntaxError) as e:
                    # These are expected parsing errors - log but continue to fallback
                    # Don't catch all exceptions, let other errors propagate
                    try:
                        from .logging_config import get_logger

                        logger = get_logger("parser")
                        logger.debug(
                            f"Failed to parse function call parts: {e}, expr_part={expr_part!r}, var_part={var_part!r}"
                        )
                    except ImportError:
                        pass
                    # If special handling fails, fall through to normal parsing
                    pass
                except Exception:
                    # Unexpected errors - re-raise them
                    raise

    # Try pattern matching on restored expression
    match = func_pattern.match(expr_str_restored.strip())
    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        # Split arguments
        parts = []
        current = ""
        depth = 0
        for char in args_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current:
            parts.append(current.strip())

        if len(parts) == 2:
            # Parse both parts
            expr_part = parts[0]
            var_part = parts[1]

            try:
                expr_parsed = parse_expr(
                    expr_part,
                    local_dict=ALLOWED_SYMPY_NAMES,
                    global_dict=SAFE_GLOBALS,
                    transformations=TRANSFORMATIONS,
                    evaluate=False,
                )
                var_parsed = parse_expr(
                    var_part,
                    local_dict=ALLOWED_SYMPY_NAMES,
                    global_dict=SAFE_GLOBALS,
                    transformations=TRANSFORMATIONS,
                    evaluate=False,
                )

                # Get the function
                func = ALLOWED_SYMPY_NAMES.get(func_name)
                if func:
                    # Call the function directly
                    result = func(expr_parsed, var_parsed)
                    # Validate the result
                    _validate_expression_tree(result)
                    return result
            except Exception:
                # If special handling fails, fall through to normal parsing
                pass

    # Normal parsing for expressions without special function call format
    # Use the restored expression (with markers replaced by commas)

    # Prepare local dictionary with allowed functions as sp.Function
    # Use provided local_dict as base if available, otherwise copy ALLOWED_SYMPY_NAMES
    local_env = local_dict.copy() if local_dict else ALLOWED_SYMPY_NAMES.copy()

    if allowed_functions:
        for name in allowed_functions:
            # We define them as UndefinedFunction (sp.Function class)
            # so they parse as proper function calls, not implicit multiplication
            # of characters (e.g. flarg -> f*l*a*r*g)
            if name not in local_env:
                local_env[name] = sp.Function(name)

    expr = parse_expr(
        expr_str_restored,
        local_dict=local_env,
        global_dict=SAFE_GLOBALS,  # SANDBOX: Prevent SymPy from injecting globals (like 'test') by default
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )
    # Validate expression tree structure
    # Allow None as a valid result (e.g., from print() which returns None)
    _validate_expression_tree(
        expr, allow_none=True, allowed_functions=allowed_functions
    )
    return expr


def format_inequality_solution(sol_str: str) -> str:
    """Format SymPy inequality solution string for readability.

    Converts complex inequality representations to more readable forms.
    Handles compound inequalities like "a < x < b".

    Args:
        sol_str: Raw inequality solution string from SymPy

    Returns:
        Formatted inequality string
    """
    pattern = re.compile(
        r"\((.*?)\s*([<>=!]+)\s*(.*?)\)\s*&\s*\((.*?)\s*([<>=!]+)\s*(.*?)\)"
    )
    match = pattern.match(sol_str)
    if not match:
        return sol_str
    groups = [group.strip() for group in match.groups()]
    expr1, op1, var1, expr2, op2, var2 = groups
    if var1 == expr2:
        if op1 in ("<", "<=") and op2 in ("<", "<="):
            return f"{expr1} {op1} {var1} {op2} {var2}"
    elif expr1 == var2:
        op_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
        if op1 in op_map and op2 in op_map:
            return f"{var1} {op_map[op1]} {expr1} {op_map[op2]} {var2}"
    return sol_str


def split_top_level_commas(input_str: str) -> list[str]:
    """Split string by commas that are not inside (), [], {}, or quotes."""
    parts: list[str] = []
    current = []
    depth_paren = depth_brack = depth_brace = 0
    in_quote = False
    quote_char = None

    i = 0
    while i < len(input_str):
        char = input_str[i]

        # Handle quotes
        if char in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                # check for non-escaped quote
                if i > 0 and input_str[i - 1] == "\\":
                    pass  # escaped
                else:
                    in_quote = False
                    quote_char = None

        # Handle split condition
        if (
            char == ","
            and not in_quote
            and depth_paren == 0
            and depth_brack == 0
            and depth_brace == 0
        ):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 1
            continue

        # Handle brackets (only if not in quote)
        if not in_quote:
            if char == "(":
                depth_paren += 1
            elif char == ")":
                depth_paren = max(0, depth_paren - 1)
            elif char == "[":
                depth_brack += 1
            elif char == "]":
                depth_brack = max(0, depth_brack - 1)
            elif char == "{":
                depth_brace += 1
            elif char == "}":
                depth_brace = max(0, depth_brace - 1)

        current.append(char)
        i += 1

    # append last segment
    last = "".join(current).strip()
    if last:
        parts.append(last)
    return parts


def expand_function_calls(expr_str: str) -> str:
    """Expand function calls in an expression string.

    This function finds all function calls (e.g., f(2), g(x,y)) and replaces
    them with their evaluated values if the functions are defined.

    Uses a recursive approach to handle nested function calls properly.

    Args:
        expr_str: Expression string that may contain function calls

    Returns:
        Expression string with function calls expanded (if functions are defined)

    Raises:
        ValidationError: If a function call has wrong number of arguments (WRONG_ARGUMENT_COUNT)
    """
    try:
        from .function_manager import evaluate_function
        from .function_manager import list_functions
        from .function_manager import parse_function_call
        from .types import ValidationError

        # If no functions are defined, return original
        defined_funcs = list_functions()
        if not defined_funcs:
            return expr_str

        # Find function calls using a recursive approach
        def find_and_replace_calls(s: str, start_pos: int = 0) -> tuple[str, int]:
            """Recursively find and replace function calls.

            Returns:
                (modified_string, new_position)
            """
            result_parts = []
            i = start_pos

            while i < len(s):
                # Look for function name pattern followed by (
                if i < len(s) - 1 and s[i].isalpha():
                    # Find the function name
                    func_start = i
                    while i < len(s) and (s[i].isalnum() or s[i] == "_"):
                        i += 1
                    func_name = s[func_start:i]

                    # Skip whitespace
                    while i < len(s) and s[i].isspace():
                        i += 1

                    # Check if this is followed by (
                    if i < len(s) and s[i] == "(" and func_name in defined_funcs:
                        # Found a potential function call
                        result_parts.append(s[start_pos:func_start])

                        # Parse the function call
                        i += 1  # Skip '('
                        args_str, new_i = parse_args(s, i)
                        i = new_i

                        if i < len(s) and s[i] == ")":
                            # Valid function call
                            i += 1  # Skip ')'

                            # Parse and evaluate
                            func_call = parse_function_call(
                                func_name + "(" + args_str + ")"
                            )
                            if func_call:
                                call_func_name, arg_strings = func_call

                                # Recursively expand arguments (may contain nested calls)
                                expanded_args = []
                                for arg_str in arg_strings:
                                    # Expand nested function calls in argument
                                    expanded_arg, _ = find_and_replace_calls(
                                        arg_str.strip(), 0
                                    )
                                    # Parse the argument directly (avoid recursion by not using parse_preprocessed)
                                    try:
                                        # Use basic SymPy parsing without going through preprocess
                                        # which would call expand_function_calls again
                                        arg_expr = parse_expr(
                                            expanded_arg,
                                            local_dict=ALLOWED_SYMPY_NAMES,
                                            global_dict=SAFE_GLOBALS,
                                            transformations=TRANSFORMATIONS,
                                            evaluate=True,
                                        )
                                        expanded_args.append(arg_expr)
                                    except Exception:
                                        # If parsing fails, treat as symbol
                                        expanded_args.append(sp.Symbol(expanded_arg))

                                # Evaluate function
                                try:
                                    result = evaluate_function(
                                        call_func_name, expanded_args
                                    )
                                    result_parts.append("(" + str(result) + ")")
                                    start_pos = i
                                    continue
                                except ValidationError as ve:
                                    # If it's a wrong argument count error, propagate it
                                    # This ensures users get clear error messages
                                    if ve.code == "WRONG_ARGUMENT_COUNT":
                                        raise  # Re-raise so it can be caught and displayed properly
                                    # For other validation errors, keep original
                                    result_parts.append(
                                        func_name + "(" + args_str + ")"
                                    )
                                    start_pos = i
                                    continue
                                except Exception:
                                    # Evaluation failed for other reasons, keep original
                                    result_parts.append(
                                        func_name + "(" + args_str + ")"
                                    )
                                    start_pos = i
                                    continue
                        else:
                            # Not a valid function call, keep original
                            result_parts.append(s[start_pos:func_start])
                            start_pos = func_start
                            # Skip the whole function name and continue
                            i = func_start + len(func_name)
                            continue
                    else:
                        # Not a function call, continue
                        # We've already advanced i past the identifier, so just continue
                        continue
                else:
                    i += 1

            result_parts.append(s[start_pos:])
            return "".join(result_parts), len(s)

        def parse_args(s: str, start: int) -> tuple[str, int]:
            """Parse arguments inside parentheses, handling nested parentheses.

            Returns:
                (args_string, new_position)
            """
            args = []
            current = []
            depth = 0
            i = start

            while i < len(s):
                if s[i] == "(":
                    depth += 1
                    current.append(s[i])
                elif s[i] == ")":
                    if depth == 0:
                        break
                    depth -= 1
                    current.append(s[i])
                elif s[i] == "," and depth == 0:
                    args.append("".join(current))
                    current = []
                else:
                    current.append(s[i])
                i += 1

            if current:
                args.append("".join(current))

            args_str = ",".join(args)
            return args_str, i

        result, _ = find_and_replace_calls(expr_str, 0)
        return result

    except ImportError:
        # function_manager not available, return original
        return expr_str
    except ValidationError:
        # Re-raise ValidationError (especially WRONG_ARGUMENT_COUNT) so it can be displayed
        raise
    except Exception:
        # Any other error, return original
        return expr_str
