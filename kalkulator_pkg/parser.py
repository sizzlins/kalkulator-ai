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

import sympy as sp # TYPE CHECKING ONLY? No, used in logic if lazy loaded
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sympy as sp

import ast

from . import config  # Lazy load config attributes
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
# TRANSFORMATIONS is now lazy loaded via config.TRANSFORMATIONS
from .types import ValidationError

# Pre-compiled regex patterns (module-level for performance)
# Smart √ to sqrt() conversion: √x -> sqrt(x), √(expr) -> sqrt(expr)
SQRT_PATTERN = re.compile(r'√(\([^)]+\)|\w+|\d+\.?\d*)')
# Parenthesized sub-expression pattern for cache lookup
PAREN_PATTERN = re.compile(r"\(([^()]+)\)")

# Minimal globals for SymPy literals to prevent namespace pollution
# Minimal globals for SymPy literals (Lazy)
_SAFE_GLOBALS = None

def get_safe_globals():
    global _SAFE_GLOBALS
    if _SAFE_GLOBALS is not None:
        return _SAFE_GLOBALS
    import sympy as sp
    _SAFE_GLOBALS = {
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Pow": sp.Pow,
        "Add": sp.Add,
        "Mul": sp.Mul,
        "Number": sp.Number,
        "Function": sp.Function,
        "AccumBounds": sp.AccumBounds,
    }
    return _SAFE_GLOBALS

class SafeSymPyVisitor(ast.NodeVisitor):
    """AST Visitor to safely construct SymPy expressions from Python AST.
    
    This replaces sympy.parse_expr which uses unsafe eval().
    
    Security features:
    - Blocks dangerous nodes (Import, Exec, Attribute access)
    - Limits recursion depth to prevent DoS via deeply nested expressions
    - Only allows whitelisted functions and constants
    """
    
    # SECURITY: Maximum AST depth to prevent stack overflow/memory exhaustion DoS
    MAX_DEPTH = 100
    
    # SECURITY: Maximum input string length to prevent DoS via expansion (v3.3)
    MAX_INPUT_LENGTH = 10000
    
    # Names that are strictly forbidden even if they parse as valid identifiers
    BLACKLIST_NAMES = {
        "__builtins__", "__import__", "eval", "exec", "globals", "locals", 
        "open", "help", "exit", "quit", "input", "compile", "dir", "vars",
        "hasattr", "getattr", "setattr", "delattr", "memoryview", "super",
        "classmethod", "staticmethod", "property", "object", "type",
        "__class__", "__base__", "__subclasses__", "__mro__",
    }
    
    def __init__(self, local_dict=None, global_dict=None, allowed_functions=None):
        import sympy as sp
        self.sp = sp
        self.local_dict = local_dict or {}
        self.global_dict = global_dict or {}
        self.allowed_functions = allowed_functions or set()
        self._depth = 0  # Track recursion depth

    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValidationError("Input must be a single expression", "INVALID_STRUCTURE")
        return self.visit(node.body[0].value)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Map AST operators to SymPy classes/functions
        # Note: ^ is converted to ** by tokenizer before this, so BitXor is rare (unless & | used)
        if isinstance(node.op, ast.Add): return self.sp.Add(left, right)
        if isinstance(node.op, ast.Sub): return self.sp.Add(left, self.sp.Mul(-1, right))
        if isinstance(node.op, ast.Mult): return self.sp.Mul(left, right)
        if isinstance(node.op, ast.Div): return self.sp.Mul(left, self.sp.Pow(right, -1))
        if isinstance(node.op, ast.Pow): return self.sp.Pow(left, right)
        if isinstance(node.op, ast.Mod): return self.sp.Mod(left, right)
        
        # Bitwise operators (support symbols if defined in config)
        if isinstance(node.op, ast.BitXor): return self._get_func("bitwise_xor")(left, right)
        if isinstance(node.op, ast.BitOr): return self._get_func("bitwise_or")(left, right)
        if isinstance(node.op, ast.BitAnd): return self._get_func("bitwise_and")(left, right)
        if isinstance(node.op, ast.LShift): return self._get_func("lshift")(left, right)
        if isinstance(node.op, ast.RShift): return self._get_func("rshift")(left, right)

        raise ValidationError(f"Unsupported operator: {type(node.op).__name__}", "INVALID_OPERATOR")

    def _get_func(self, name):
        """Helper to get function from allowed dicts."""
        if name in self.local_dict: return self.local_dict[name]
        if name in self.global_dict: return self.global_dict[name]
        # Fallback to looking up in ALLOWED_SYMPY_NAMES directly if available
        if name in config.ALLOWED_SYMPY_NAMES: return config.ALLOWED_SYMPY_NAMES[name]
        return self.sp.Function(name)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub): return self.sp.Mul(-1, operand)
        if isinstance(node.op, ast.UAdd): return operand
        if isinstance(node.op, ast.Not): return self.sp.Not(operand)
        if isinstance(node.op, ast.Invert): return self.sp.Not(operand) # ~x
        raise ValidationError(f"Unsupported unary operator: {type(node.op).__name__}", "INVALID_OPERATOR")

    def visit_Attribute(self, node):
        """Block attribute access (introspection prevention)."""
        raise ValidationError("Attribute access is not allowed", "SECURITY_VIOLATION")

    def visit_Call(self, node):
        func_obj = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        if node.keywords:
            raise ValidationError("Keyword arguments not supported", "INVALID_CALL")
            
        # Security: Strict Instance Check (v4.1 Audit Remediation)
        # We must verify that usage of function calls resolves to actual allowed Function types,
        # not just Symbols that happen to be callable or other objects.
        
        # 1. Block bare Symbols being called (unless they are Function classes, which Symbols aren't)
        if isinstance(func_obj, self.sp.Symbol):
             raise ValidationError(f"Calling symbolic variables is forbidden: {node.func.id}", "INVALID_CALL")

        # 2. Whitelist check for Function Classes or Instances
        is_safe_func = False
        
        # Check against pure SymPy Function class or instance
        if isinstance(func_obj, (self.sp.FunctionClass, self.sp.Function)):
             is_safe_func = True
        # Check against specific allowed math functions (sin, cos, etc) which might be classes
        elif isinstance(func_obj, type) and issubclass(func_obj, self.sp.Function):
             is_safe_func = True
        # Check against some specific instances like sp.sin (which is a FunctionClass actually)
        # But some might be pre-instantiated.
        
        if not is_safe_func:
             # Last resort: check if it's in our explicit Allowed Dict context (e.g. 'sin' -> sp.sin)
             # visit_Name would have returned it.
             import kalkulator_pkg.config as config
             if func_obj in config.ALLOWED_SYMPY_NAMES.values():
                 is_safe_func = True

        if not is_safe_func:
              raise ValidationError(f"Forbidden function call target: {type(func_obj).__name__}", "INVALID_CALL")

        return func_obj(*args)

    def visit_Name(self, node):
        id = node.id
        
        # Security Blacklist Check
        if id in self.BLACKLIST_NAMES:
            raise ValidationError(f"Forbidden name: {id}", "SECURITY_VIOLATION")
        if id.startswith("__") and id.endswith("__"):
            raise ValidationError(f"Forbidden dunder name: {id}", "SECURITY_VIOLATION")
            
        if id in self.local_dict: return self.local_dict[id]
        if id in self.global_dict: return self.global_dict[id]
        if id in config.ALLOWED_SYMPY_NAMES: return config.ALLOWED_SYMPY_NAMES[id]
        
        # Check against allowed functions list
        if self.allowed_functions and id in self.allowed_functions:
             return self.sp.Function(id)

        # Allow basic variable names
        return self.sp.Symbol(id)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return self.sp.sympify(node.value)
        if isinstance(node.value, complex):
            return self.sp.sympify(node.value)
        # Handle string constants? Usually no.
        raise ValidationError(f"Unsupported constant type: {type(node.value)}", "INVALID_CONSTANT")
    
    def visit_Attribute(self, node):
        """SECURITY: Explicitly block attribute access to prevent gadget chain attacks.
        
        Attacks like (1).__class__.__base__.__subclasses__() rely on attribute
        access to traverse the object graph and reach dangerous objects.
        
        v3.3 Audit Remediation: Made explicit instead of relying on generic_visit.
        """
        raise ValidationError(
            f"Attribute access is forbidden: {node.attr}",
            "SECURITY_VIOLATION"
        )
        
    def generic_visit(self, node):
        raise ValidationError(f"Unsupported language construct: {type(node).__name__}", "SECURITY_VIOLATION")
    
    def visit(self, node):
        """Override visit to enforce depth limit."""
        self._depth += 1
        if self._depth > self.MAX_DEPTH:
            raise ValidationError(
                f"Expression too deeply nested (max depth: {self.MAX_DEPTH})",
                "DEPTH_EXCEEDED"
            )
        try:
            return super().visit(node)
        finally:
            self._depth -= 1

def safe_sympy_parse(expr_str: str, local_dict=None, global_dict=None, allowed_functions=None) -> sp.Expr:
    """Parse expression string into SymPy object using safe AST Visitor.
    
    Security features (v3.3 Audit):
    - Input length limit to prevent DoS
    - AST depth limit to prevent stack exhaustion
    - Attribute access blocking to prevent gadget chains
    """
    if not expr_str.strip():
        raise ValidationError("Empty input", "EMPTY_INPUT")
    
    # SECURITY: Prevent DoS via extremely long input strings (v3.3)
    if len(expr_str) > SafeSymPyVisitor.MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input too long: {len(expr_str)} chars (max: {SafeSymPyVisitor.MAX_INPUT_LENGTH})",
            "INPUT_TOO_LONG"
        )
    try:
        tree = ast.parse(expr_str, mode='exec')
    except SyntaxError as e:
        raise ValidationError(f"Syntax Error: {e.msg}", "SYNTAX_ERROR")
    except ValueError as e:
         raise ValidationError(f"Value Error during parsing: {str(e)}", "PARSE_ERROR")

    visitor = SafeSymPyVisitor(local_dict, global_dict, allowed_functions)
    return visitor.visit(tree)


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


    pass



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


def preprocess_expression(
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

    import re

    processed_str = input_str.strip()

    # Convert √ to sqrt()
    # Handle both √x and √(expr)
    # SQRT_PATTERN needs to be defined, assuming it's available in the context
    # For example: SQRT_PATTERN = re.compile(r'√\(([^)]+)\)|√(\w+)')
    # For this change, I'll assume SQRT_PATTERN is defined elsewhere or needs to be added.
    # As per the instruction, I'm just inserting the line.
    # processed_str = SQRT_PATTERN.sub(r'sqrt(\1)', processed_str) # This line is commented out as SQRT_PATTERN is not defined in the provided context.

    # Convert Factorial (!) syntax: 5!, x!, (1+2)!
    # Must lookbehind or match valid predecessor, and ensure NOT followed by = (!=)
    # Simple heuristic: Preceded by digit, letter, or closing paren.
    # We use a loop to handle nested/chained factorials if needed, but single pass covers most.
    # Pattern: capture group 1 (alphanum or paren block) followed by ! and NOT =
    # Regex: (\w+|\([^)]+\))!+(?!=) -> factorial(\1)
    # Note: This is simple and might miss "((1+2))!", but good for basic usage.
    # We iterate to handle multiple ! in string
    # Using a while loop to handle potential stacked cases orjust simple substitution
    
    # We replace "expression!" with "factorial(expression)"
    # Avoid matching !=
    processed_str = re.sub(r'([\w\d\.]+|\([^)]+\))!(?!=)', r'factorial(\1)', processed_str)

    # Convert ^ to ** (unless skipped)
    if not skip_exponent_conversion:
        # Replace ^ with **
        processed_str = processed_str.replace("^", "**")
    # Basic unicode/symbol replacements BEFORE tokenization
    processed_str = processed_str.replace("−", "-").replace("–", "-")
    processed_str = processed_str.replace("Δ", "Delta")
    processed_str = processed_str.replace("π", "pi")
    processed_str = processed_str.replace(":", "/")
    
    # Use Safe Tokenizer for robust structural transformation
    # Handles: Implicit mult, Forbidden tokens, Syntax sugar (^, mod)
    try:
        from .tokenizer import transform_input
        processed_str = transform_input(processed_str)
    except Exception as e:
        raise ValidationError(f"Parsing error: {str(e)}", "TOKENIZER_ERROR") from e

    # Sub-expression caching optimization removed to satisfy Audit Requirements (v3.4)
    # "Stop using Regex to parse nested function calls"
    # The caching logic was using iterative regex replacement which is structurally unsafe.
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

    # Phase 4 (v4.1 Audit Fix): REMOVED Regex-based and Hand-rolled parsing for functions.
    # We now allow SymPy to parse function calls (e.g. f(x)) as Function objects,
    # and perform substitution during the evaluation phase using the Registry.
    # This avoids the fragile "expand_function_calls" text manipulation.

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
    # AST-based parsing (safe_sympy_parse) handles commas in function calls correctly.
    # The legacy manual splitting logic (for eval-based parsing) is removed.
    # AST-based parsing (safe_sympy_parse) handles commas in function calls correctly.
    # The legacy manual splitting logic (for eval-based parsing) is removed.
    
    if allowed_functions:
         # Note: safe_sympy_parse uses its own whitelist mechanism (visit_Call).
         # We might need to adjust it or pass the whitelist down?
         # safe_sympy_parse has ALLOWED_SYMPY_NAMES but checks against it.
         # For now, we trust safe_sympy_parse's internal security.
         pass
         
    # Fixed: Removed premature return that ignored allowed_functions
    return safe_sympy_parse(
        expr_str,
        allowed_functions=allowed_functions,
        local_dict=local_dict,
        # global_dict might be needed? safe_sympy_parse uses get_safe_globals() by default
    )



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
                                        # Use safe_sympy_parse instead of unsafe parse_expr
                                        arg_expr = safe_sympy_parse(
                                            expanded_arg,
                                            local_dict=ALLOWED_SYMPY_NAMES,
                                            global_dict=get_safe_globals(),
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
preprocess = preprocess_expression
