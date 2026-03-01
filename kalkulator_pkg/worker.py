from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sympy as sp

import sympy as sp

from .config import CACHE_SIZE_SOLVE
from .config import ENABLE_PERSISTENT_WORKER
from .config import WORKER_AS_MB
from .config import WORKER_CPU_SECONDS
from .config import WORKER_POOL_SIZE
from .config import WORKER_TIMEOUT
from .parser import parse_preprocessed
from .types import ValidationError

try:
    from .logging_config import get_logger

    logger = get_logger("worker")
except ImportError:
    # Fallback if logging not available
    class NullLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    logger = NullLogger()


class SafeJSONEncoder(json.JSONEncoder):
    """Encodes complex numbers and SymPy objects safely to strings."""
    def default(self, obj):
        if isinstance(obj, complex):
            return str(obj)
        # Handle SymPy objects
        if hasattr(obj, 'evalf'):
            try:
                # Force numerical evaluation to avoid returning raw symbolic structures
                # which can reveal the underlying function (e.g. (-15)^(15/16)).
                # N() or evalf() returns a SymPy Number (Float or Complex).
                val = obj.evalf()
                return str(val)
            except Exception:
                pass
        
        # Handle SymPy infinity/complex infinity (zoo)
        if hasattr(obj, 'is_infinite') and obj.is_infinite:
             return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            # Fallback for any other non-serializable objects
            return str(obj)

HAS_RESOURCE = False
try:
    import resource  # noqa: F401 - check if available

    HAS_RESOURCE = True
except (ImportError, OSError):
    HAS_RESOURCE = False

try:
    from multiprocessing import Event
    from multiprocessing import Manager
    from multiprocessing import Process
    from multiprocessing import Pipe
    from multiprocessing.connection import Connection
except ImportError:
    # Multiprocessing not supported or restricted
    Process = None  # type: ignore
    Pipe = None # type: ignore
    Event = None  # type: ignore
    Manager = None  # type: ignore
    Connection = None # type: ignore


def _try_numeric_integration_fallback(expr: Any) -> Any:
    """Attempt numerical integration for failed symbolic Integrals.
    
    Args:
        expr: The SymPy expression containing Integral(s).
        
    Returns:
        Expression with Integrals evaluated numerically if possible, else original.
    """
    if not isinstance(expr, sp.Basic):
         return expr
         
    # Find all top-level Integral atoms (iterative replacement)
    integrals = list(expr.atoms(sp.Integral))
    if not integrals:
        return expr
        
    replacements = {}
    for integral in integrals:
        # Only handle definite integrals (limits have 3 elements: (x, a, b))
        if not all(len(lim) == 3 for lim in integral.limits):
            continue
            
        try:
            # 1. Try mpmath via evalf() first (Symbolic numeric)
            # This handles high precision if needed
            val = integral.evalf()
            
            # Check for failure (evalf returns the Integral itself if failed)
            if isinstance(val, sp.Integral):
                raise ValueError("evalf failed")
                
            replacements[integral] = val
            
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            # 2. Fallback to SciPy (Robust numeric)
            # Useful for singularities where mpmath might struggle or be slow
            try:
                import scipy.integrate as scipy_integrate
                
                # We can only handle single integrals for now easily
                if len(integral.limits) != 1:
                    continue
                    
                limit = integral.limits[0]
                var, a, b = limit
                integrand = integral.function
                
                # Convert integrand to lambda
                f_lambda = sp.lambdify([var], integrand, modules=['numpy', 'math'])
                
                # Convert limits to float
                a_float = float(a)
                b_float = float(b)
                
                # Heuristic: Check for singularity at 0 if range includes it
                # Logic: If sign(a) != sign(b), 0 is in between.
                points = []
                if a_float < 0 < b_float:
                     points = [0]
                     
                # Execute QUADPACK
                val_float, err = scipy_integrate.quad(f_lambda, a_float, b_float, points=points)
                
                # Return as SymPy Float
                replacements[integral] = sp.Float(val_float)
                
            except Exception:
                # If SciPy also fails, keep original
                pass
                
    if replacements:
        return expr.subs(replacements)
    return expr


class WindowsJobObject:
    """Context Manager for Windows Job Objects to enforce resource limits.
    
    Implements the Supervisor Pattern: expected to be used by the PARENT process
    to manage limits for child worker processes.
    """
    def __init__(self):
        self.handle = None
        self._kernel32 = None
        self._wintypes = None
        self._ctypes = None
        
        if sys.platform == 'win32':
            try:
                import ctypes
                from ctypes import wintypes
                self._ctypes = ctypes
                self._wintypes = wintypes
                self._kernel32 = ctypes.windll.kernel32
                
                # Define signatures
                self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
                self._kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
                
                self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
                self._kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                
                self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
                self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

                self._kernel32.OpenProcess.restype = wintypes.HANDLE
                self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                
                self._kernel32.CloseHandle.restype = wintypes.BOOL
                self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                
            except ImportError:
                pass

    def __enter__(self):
        if sys.platform != 'win32' or not self._kernel32:
            return self
            
        try:
            # Constants
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
            JOB_OBJECT_LIMIT_PROCESS_TIME = 0x2
            JobObjectExtendedLimitInformation = 9
            
            # Structs
            ctypes = self._ctypes
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [('ReadOperationCount', ctypes.c_ulonglong),
                            ('WriteOperationCount', ctypes.c_ulonglong),
                            ('OtherOperationCount', ctypes.c_ulonglong),
                            ('ReadTransferCount', ctypes.c_ulonglong),
                            ('WriteTransferCount', ctypes.c_ulonglong),
                            ('OtherTransferCount', ctypes.c_ulonglong)]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                            ('PerJobUserTimeLimit', ctypes.c_longlong),
                            ('LimitFlags', ctypes.c_ulong),
                            ('MinimumWorkingSetSize', ctypes.c_size_t),
                            ('MaximumWorkingSetSize', ctypes.c_size_t),
                            ('ActiveProcessLimit', ctypes.c_ulong),
                            ('Affinity', ctypes.c_size_t),
                            ('PriorityClass', ctypes.c_ulong),
                            ('SchedulingClass', ctypes.c_ulong)]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                            ('IoInfo', IO_COUNTERS),
                            ('ProcessMemoryLimit', ctypes.c_size_t),
                            ('JobMemoryLimit', ctypes.c_size_t),
                            ('PeakProcessMemoryUsed', ctypes.c_size_t),
                            ('PeakJobMemoryUsed', ctypes.c_size_t)]
            
            # 1. Create Job Object
            self.handle = self._kernel32.CreateJobObjectW(None, None)
            if not self.handle:
                logger.warning("Failed to create Windows Job Object")
                return self
                
            # 2. Configure Limits
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_PROCESS_TIME
            
            # Memory Limit
            info.ProcessMemoryLimit = int(WORKER_AS_MB) * 1024 * 1024
            
            # CPU Limit (100ns units)
            info.BasicLimitInformation.PerProcessUserTimeLimit = int(WORKER_CPU_SECONDS * 10_000_000)
            
            if not self._kernel32.SetInformationJobObject(
                self.handle, 
                JobObjectExtendedLimitInformation, 
                ctypes.byref(info), 
                ctypes.sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
            ):
                logger.warning("Failed to set Job Object information")
                self._kernel32.CloseHandle(self.handle)
                self.handle = None
                return self
                
            logger.debug("Windows Job Object created successfully")
            
        except Exception as e:
            logger.warning(f"Windows Job Object error: {e}")
            if self.handle:
                try:
                    self._kernel32.CloseHandle(self.handle)
                except Exception:
                    pass
                self.handle = None
        
        return self

    def assign_process(self, pid: int) -> bool:
        """Assign a process (by PID) to the job object."""
        if not self.handle or not self._kernel32:
            return False
            
        # PROCESS_SET_QUOTA (0x0100) | PROCESS_TERMINATE (0x0001) required for job object assignment
        # Using PROCESS_ALL_ACCESS for simplicity but could be tightened
        PROCESS_ALL_ACCESS = 0x1F0FFF
        
        try:
            hProcess = self._kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            if not hProcess:
                logger.debug(f"Could not open process {pid} for Job assignment")
                return False
                
            success = self._kernel32.AssignProcessToJobObject(self.handle, hProcess)
            self._kernel32.CloseHandle(hProcess)
            
            if success:
                logger.debug(f"Assigned process {pid} to Job Object")
                return True
            else:
                logger.debug(f"Failed to assign process {pid} to Job Object (Error)")
                return False
        except Exception as e:
            logger.warning(f"Error assigning process to Job Object: {e}")
            return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle and self._kernel32:
            try:
                self._kernel32.CloseHandle(self.handle)
                logger.debug("Closed Windows Job Object handle")
            except Exception as e:
                logger.warning(f"Error closing Job Object handle: {e}")
            self.handle = None



def _limit_resources() -> None:
    """Apply resource limits (Unix `resource` or Windows Job Objects)."""
    # 1. Unix Logic
    if HAS_RESOURCE:
        try:
            import resource as _resource
            _resource.setrlimit(
                _resource.RLIMIT_CPU, (int(WORKER_CPU_SECONDS), int(WORKER_CPU_SECONDS) + 1)
            )
            _resource.setrlimit(
                _resource.RLIMIT_AS,
                (int(WORKER_AS_MB) * 1024 * 1024, int(WORKER_AS_MB) * 1024 * 1024 + 1),
            )
        except (ImportError, OSError, ValueError):
            pass

    # 2. Windows Logic
    # 2. Windows Logic
    # Handled by WindowsJobObject context manager in _worker_daemon_main
    pass



def _format_evaluation_result(expr: sp.Basic) -> str | int | float | complex:
    """Format a SymPy expression result as a number or string.

    v3.4 Audit Remediation: Returns actual numbers (int/float/complex) for numeric 
    results instead of strings like "0" or "3.14", improving type stability.
    Symbolic expressions are still returned as strings.

    Args:
        expr: SymPy expression to format

    Returns:
        int/float/complex for numbers, str for symbolic expressions
    """
    # Try numeric evaluation first for exact results
    try:
        # Standardize: simplify first if it's simple
        if not expr.free_symbols:
            # Check for exact integer
            if expr.is_integer:
                 return int(expr)
            
            # Use full precision eval
            num_val = sp.N(expr, 20)
            
            if hasattr(num_val, "is_number") and num_val.is_number:
                # Check complex
                try:
                    from .config import NUMERIC_TOLERANCE, SNAPPING_THRESHOLD
                    c_val = complex(num_val)
                    # Check if imaginary part is negligible (using standard tolerance)
                    if abs(c_val.imag) < NUMERIC_TOLERANCE:
                        # Real
                        r_val = float(c_val.real)
                        # Check integer (using strict snapping threshold to preserve small numbers like 1e-14)
                        if abs(r_val - round(r_val)) < SNAPPING_THRESHOLD:
                            return int(round(r_val))
                        return r_val
                    else:
                        return c_val
                except Exception:
                    pass
    except Exception:
        # Formatting failed, fall back to string
        pass
        
    # Fallback to string for symbolic
    return str(expr)




def _get_user_friendly_error_message(
    error: Exception, input_str: str
) -> tuple[str, str]:
    """Generate user-friendly error messages for common errors.

    Returns:
        Tuple of (error_message, error_code)
    """
    error_type = type(error).__name__
    error_msg = str(error)
    input_stripped = input_str.strip()

    # Check for common single-character operator errors
    if input_stripped in ["-", "+", "*", "/", "^", "%", "="]:
        return (
            f"'{input_stripped}' is an operator, not a complete expression. "
            f"Use it in an expression like '5{input_stripped}3' or 'x{input_stripped}2'.",
            "INCOMPLETE_EXPRESSION",
        )

    # Check for empty or whitespace-only input
    if not input_stripped or input_stripped.isspace():
        return (
            "Empty input. Please enter a valid expression, equation, or command.",
            "EMPTY_INPUT",
        )

    # Check for backslash at end (line continuation character)
    if len(input_stripped) > 0 and input_stripped[-1] == "\\":
        return (
            "Expression ends with '\\' (backslash), which is a line continuation character. "
            "Remove the backslash or complete the expression on the next line.",
            "INCOMPLETE_EXPRESSION",
        )

    # Check for unterminated expressions (ends with operator)
    if len(input_stripped) > 0 and input_stripped[-1] in [
        "-",
        "+",
        "*",
        "/",
        "^",
        "%",
        "=",
    ]:
        return (
            f"Expression ends with '{input_stripped[-1]}'. "
            f"Complete the expression, for example: '5{input_stripped[-1]}3' or 'x{input_stripped[-1]}2'.",
            "INCOMPLETE_EXPRESSION",
        )

    # Check for TokenError (from tokenize module) - often indicates syntax issues like backslash
    error_type_name = type(error).__name__
    if "TokenError" in error_type_name:
        if (
            "unexpected EOF" in error_msg.lower()
            or "multi-line statement" in error_msg.lower()
        ):
            # Check if input ends with backslash
            if len(input_stripped) > 0 and input_stripped[-1] == "\\":
                return (
                    "Expression ends with '\\' (backslash), which is a line continuation character. "
                    "Remove the backslash or complete the expression on the next line.",
                    "INCOMPLETE_EXPRESSION",
                )
            return (
                "Incomplete expression: Backslash '\\' at the end indicates line continuation. "
                "Remove the backslash or complete the expression on the next line.",
                "INCOMPLETE_EXPRESSION",
            )

    # Check for SyntaxError with specific patterns
    if isinstance(error, SyntaxError):
        if "unexpected EOF" in error_msg.lower() or "EOF" in error_msg:
            # Check if it's specifically about multi-line statement (backslash issue)
            if "multi-line statement" in error_msg.lower():
                return (
                    "Incomplete expression: Backslash '\\' at the end indicates line continuation. "
                    "Remove the backslash or complete the expression on the next line.",
                    "INCOMPLETE_EXPRESSION",
                )
            return (
                "Incomplete expression. Check for missing operands, unmatched parentheses, or unterminated strings.",
                "SYNTAX_ERROR",
            )
        if "leading zeros" in error_msg.lower() or "0o prefix" in error_msg.lower():
            # Check if input looks like a hexadecimal number
            input_clean = input_str.strip()
            # Look for patterns like "123edc09f2"
            import re

            hex_pattern = re.compile(r"[0-9a-fA-F]{4,}")
            if hex_pattern.search(input_clean):
                return (
                    f"Invalid number format: '{input_clean}'. "
                    f"If this is a hexadecimal number, use '0x' prefix: '0x{input_clean}'. "
                    f"Otherwise, check for invalid leading zeros in decimal numbers.",
                    "SYNTAX_ERROR",
                )
            return (
                "Invalid number format: Leading zeros are not allowed in decimal integers. "
                "Use 0x prefix for hexadecimal numbers (e.g., 0x09), or remove leading zeros from decimal numbers.",
                "SYNTAX_ERROR",
            )
        if "invalid syntax" in error_msg.lower():
            # Check if error mentions leading zeros (hex number issue)
            if "leading zeros" in error_msg.lower():
                # Check if input looks like a hexadecimal number
                input_clean = input_str.strip()
                import re

                hex_pattern = re.compile(r"[0-9a-fA-F]{4,}")
                if hex_pattern.search(input_clean):
                    return (
                        f"Invalid number format: '{input_clean}' looks like a hexadecimal number. "
                        f"Use '0x' prefix: '0x{input_clean}'.",
                        "SYNTAX_ERROR",
                    )
            # Try to extract position information
            if hasattr(error, "offset") and error.offset:
                pos = error.offset
                if pos <= len(input_str):
                    char_at_pos = input_str[pos - 1 : pos] if pos > 0 else ""
                    return (
                        f"Invalid syntax at position {pos} (character '{char_at_pos}'). "
                        f"Check for typos, missing operators, or incorrect function syntax.",
                        "SYNTAX_ERROR",
                    )
            return (
                "Invalid syntax. Check for typos, missing operators, unmatched parentheses, or incorrect function calls.",
                "SYNTAX_ERROR",
            )

    # Check for ValueError with specific patterns
    if isinstance(error, ValueError):
        if "cannot assign" in error_msg.lower():
            return (
                "Cannot use '=' for assignment in this context. "
                "For equations, use '==' (double equals). For variable assignments, use separate statements.",
                "PARSE_ERROR",
            )
        if "invalid" in error_msg.lower() and "name" in error_msg.lower():
            return (
                "Invalid variable or function name. "
                "Names must start with a letter and contain only letters, numbers, and underscores.",
                "INVALID_NAME",
            )

    # Check for NameError (often from malformed parser output like "x1.207..." becoming "Symbol('x')*Number*...")
    if isinstance(error, NameError):
        if "Number" in error_msg or "Float" in error_msg or "Integer" in error_msg:
            return (
                "Invalid number format. Check for malformed numbers like 'x1.207' - "
                "did you mean 'x * 1.207' (multiplication) or 'x = 1.207' (assignment)?",
                "PARSE_ERROR",
            )
        return (
            f"Unknown identifier in expression: {error_msg}. "
            "Check for typos in variable or function names.",
            "PARSE_ERROR",
        )

    # Check for TokenError (unterminated strings, etc.)
    try:
        import tokenize

        if isinstance(error, tokenize.TokenError):
            if "unterminated" in error_msg.lower():
                return (
                    "Unmatched or unterminated string literal. Check that all quotes are properly closed and matched.",
                    "SYNTAX_ERROR",
                )
    except (ImportError, AttributeError):
        pass

    # Check for common parse error patterns
    if "parse" in error_msg.lower() or "PARSE_ERROR" in error_type:
        if "unexpected" in error_msg.lower():
            return (f"Unexpected token or character. {error_msg}", "PARSE_ERROR")
        if "invalid" in error_msg.lower():
            return (f"Invalid expression format. {error_msg}", "PARSE_ERROR")

    # Default error message
    return (f"{error_msg}. Please check your input syntax.", "PARSE_ERROR")


def worker_evaluate(
    preprocessed_expr: str, 
    allowed_functions: frozenset[str] | None = None,
    user_functions: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate a preprocessed expression in a sandboxed worker.
    
    Args:
        preprocessed_expr: The expression string (preprocessed).
        allowed_functions: Set of allowed function names (for parsing).
        user_functions: Dictionary of user-defined functions {name: (params, body)} for substitution.
    """
    logger.debug(f"Evaluating expression: {preprocessed_expr[:100]}...")
    if HAS_RESOURCE:
        try:
            _limit_resources()
            logger.debug("Resource limits applied")
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to apply resource limits: {e}")
            
    try:
        # Phase 4 Audit Fix: Parse first (as function calls), then substitute.
        # This replaces the insecure 'expand_function_calls' pre-parser.
        
        # Add user-defined function names to allowed_functions so parser accepts them
        combined_allowed = allowed_functions
        if user_functions:
            user_func_names = frozenset(user_functions.keys())
            if combined_allowed:
                combined_allowed = combined_allowed | user_func_names
            else:
                combined_allowed = user_func_names
        
        expr = parse_preprocessed(
            preprocessed_expr, allowed_functions=combined_allowed
        )
        
        # Substitute User Functions if provided
        if user_functions and expr is not None:
             # Look for Function atoms that match user definitions
             # We use a loop to handle nested dependencies (e.g. f(g(x)))
             # but limit iterations to avoid infinite recursion.
             # However, simple substitution should work if we rely on SymPy's subs or replace.
             
             # Create substitution map for known functions
             # But we need to handle arguments: f(2) -> f_body.subs(params, (2,))
             
             def replace_user_func(node):
                 if isinstance(node, sp.Function):
                     name = type(node).__name__ # Get the function name (e.g. "f")
                     if name in user_functions:
                         params, body = user_functions[name]
                         args = node.args
                         
                         # Validate argument count
                         if len(args) != len(params):
                             # We can't raise specific error here easily inside replace, 
                             # but we can return node (fail to sub) or raise.
                             # Raising helps the user know why.
                             raise ValidationError(
                                 f"Function '{name}' expects {len(params)} argument(s) but got {len(args)}.",
                                 "WRONG_ARGUMENT_COUNT"
                             )
                             
                         # Create substitution dict: param_symbol -> arg_expr
                         subs_pairs = [(sp.Symbol(p), arg) for p, arg in zip(params, args)]
                         subs_map = dict(subs_pairs)
                         
                         # safely substitute
                         subbed = body.subs(subs_map)
                         
                         try:
                             # Limit Snapper: If singularity (nan/zoo), try calculus limit
                             # e.g. f(0) = (0+1)^(1/0) -> 1^inf -> nan => Limit is e
                             if subbed is sp.nan or subbed is sp.zoo or getattr(subbed, "has", lambda *a: False)(sp.nan, sp.zoo):
                                 if len(params) == 1:
                                     # Attempt limit x -> arg
                                     p_sym, arg_val = subs_pairs[0]
                                     try:
                                         limit_val = sp.limit(body, p_sym, arg_val)
                                         if limit_val is not sp.nan:
                                             return limit_val
                                     except Exception:
                                         pass
                         except Exception:
                             pass
                         
                         return subbed
                 return node
                 
             # We need to handle nested calls: f(g(x)).
             # replace(query, value) works.
             expr = expr.replace(lambda x: isinstance(x, sp.Function) and type(x).__name__ in user_functions, replace_user_func)

    except ValidationError as e:
        if e.code == "SYNTAX_ERROR" or e.code == "TOKENIZER_ERROR":
            logger.debug(f"Validation error: {e.code} - {e.message}")
        else:
            logger.debug(f"Validation error: {e.code} - {e.message}")
        return {"ok": False, "error": str(e), "error_code": e.code}
    except (ValueError, SyntaxError) as e:
        # Only log at debug level since we're providing a user-friendly error message
        logger.debug(f"Parse error: {e}")
        error_msg, error_code = _get_user_friendly_error_message(e, preprocessed_expr)
        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }
    except Exception as e:
        # Check if this is a TokenError (from tokenize module) which often indicates syntax issues
        error_type_name = type(e).__name__
        is_token_error = "TokenError" in error_type_name
        
        # Use the user-friendly error message helper
        error_msg, error_code = _get_user_friendly_error_message(e, preprocessed_expr)

        # For syntax/parse/tokenize/name errors, log at debug level since we have a user-friendly message
        if isinstance(e, (SyntaxError, ValueError, NameError)) or is_token_error:
            logger.debug(f"Parse/tokenize error: {e}")
        else:
            # Log full traceback for truly unexpected errors
            logger.exception("Unexpected parse error in worker")

        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }

    # Handle None result (e.g., from print() which executes but returns None)
    if expr is None:
        return {
            "ok": True,
            "result": "None",
            "approx": None,
            "free_symbols": [],
        }

    try:

        # 1. Evaluate Integrals (Lazy -> Eager)
        if hasattr(expr, "has") and expr.has(sp.Integral):
             # Attempt symbolic evaluation first
             try:
                 # .doit() forces evaluation of lazy objects like Integral, Sum, Limit
                 # deep=True ensures nested objects are evaluated
                 expr = expr.doit(deep=True)
             except Exception:
                 pass # Keep as Integral on failure

             # If still Integral and definite, try numerical fallback
             if expr.has(sp.Integral):
                  expr = _try_numeric_integration_fallback(expr)

        # Simplify to canonical form
        # Skip simplification for containers (lists, tuples, arrays, dicts, bools) as they crash SymPy
        if isinstance(expr, (list, tuple, np.ndarray, dict, bool, set)):
            res = expr
        else:
            try:
                res = sp.simplify(expr)
            except (AttributeError, TypeError):
                # SymPy might crash on lists/objects that passed the isinstance check or were missed
                res = expr

        # Format result string with canonical numeric representation
        # This ensures sin(0) -> "0" and cos(0) -> "1" consistently
        result_str = _format_evaluation_result(res)

        free_syms = [str(s) for s in getattr(res, "free_symbols", set())]
        approx = None
        try:
            # Skip floating point approx for containers
            if isinstance(res, (list, tuple, np.ndarray, dict, bool, set)):
                approx = None
            else:
                approx_val = sp.N(res)
                approx_str = str(approx_val)
                if approx_str not in ("zoo", "oo", "-oo", "nan"):
                    approx = approx_str
        except (ValueError, TypeError, ArithmeticError, AttributeError):
            approx = None
        return {
            "ok": True,
            "result": result_str,
            "approx": approx,
            "free_symbols": free_syms,
        }
    except (ValueError, TypeError, ArithmeticError) as e:
        return {
            "ok": False,
            "error": f"Evaluation failed: {e}",
            "error_code": "EVAL_ERROR",
        }
    except Exception as e:
        # Catch-all for truly unexpected errors - log full traceback
        logger.exception("Unexpected evaluation error in worker")
        return {
            "ok": False,
            "error": f"Evaluation failed: {e}",
            "error_code": "UNKNOWN_ERROR",
        }


def _build_self_cmd(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [os.path.realpath(sys.argv[0])] + args
    else:
        return [
            sys.executable,
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "kalkulator.py")
            ),
        ] + args


def _retry_with_backoff(
    func, max_retries: int = 3, initial_delay: float = 0.1, max_delay: float = 2.0
) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: Callable that returns a result or raises an exception
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds

    Returns:
        Result from func() if successful

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_code = getattr(e, "code", None)
            # Check if error is transient (retryable)
            transient_codes = {"COMM_ERROR", "TIMEOUT", "UNKNOWN_ERROR"}
            if error_code not in transient_codes and attempt < max_retries:
                # Fatal error, don't retry
                raise
            if attempt < max_retries:
                logger.debug(
                    f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

    # All retries exhausted
    raise last_exception


class _WorkerManager:
    def __init__(self) -> None:
        self.procs: list[Process] = []
        self.worker_conns: list[Connection] = []  # Parent end of pipes
        self.stop_event = None
        self._next_idx = 0
        self._manager = None
        self.job_object = None  # Supervisor Job Object
        self._cancel_flags: dict[str, bool] | None = (
            None  # req_id -> cancel flag (shared dict)
        )

    def start(self) -> None:
        if not ENABLE_PERSISTENT_WORKER or Process is None or Manager is None:
            return
        if self.is_alive():
            return
        # Create Manager for shared state (works on Windows)
        if self._manager is None:
            self._manager = Manager()
            self._cancel_flags = self._manager.dict()
            
        # Create/Initialize Job Object immediately in supervisor (Parent Process)
        # This will hold the limits for all children
        # Create/Initialize Job Object immediately in supervisor (Parent Process)
        # This will hold the limits for all children
        self.job_object = WindowsJobObject()
        self.job_object.__enter__()
            
        self.stop_event = Event()
        self.worker_conns = []
        self.procs = []
        
        n = max(1, int(WORKER_POOL_SIZE or 1))
        for _ in range(n):
            parent_conn, child_conn = Pipe(duplex=True)
            proc = Process(
                target=_worker_daemon_main,
                args=(child_conn, self.stop_event, self._cancel_flags),
                daemon=True,
            )
            proc.start()
            
            # v3.4 Security Fix: Assign worker to Job Object from Supervisor (here)
            if self.job_object and proc.pid:
                self.job_object.assign_process(proc.pid)
                
            self.worker_conns.append(parent_conn)
            self.procs.append(proc)
            
            # Close child connection in parent process so only worker has it
            child_conn.close()

    def is_alive(self) -> bool:
        return bool(self.procs and all(p.is_alive() for p in self.procs))

    def stop(self) -> None:
        """Stop all worker processes gracefully."""
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            for p in self.procs or []:
                try:
                    p.join(timeout=1.0)
                except (OSError, ValueError, AttributeError) as e:
                    # Process already dead or invalid - log but continue cleanup
                    try:
                        from .logging_config import safe_log

                        safe_log(
                            "worker", "warning", f"Error joining worker process: {e}"
                        )
                    except ImportError:
                        pass
        except (AttributeError, TypeError) as e:
            # Invalid state - log but continue cleanup
            try:
                from .logging_config import safe_log

                safe_log("worker", "warning", f"Error stopping workers: {e}")
            except ImportError:
                pass
        finally:
            for conn in self.worker_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self.procs = []
            self.worker_conns = []
            self.stop_event = None
            if self._cancel_flags is not None:
                self._cancel_flags.clear()
            # Clean up Supervisor Job Object
            if self.job_object:
                self.job_object.__exit__(None, None, None)
                self.job_object = None

    def cancel_request(self, req_id: str) -> bool:
        """Cancel a pending request by ID. Returns True if cancellation flag was found."""
        if self._cancel_flags is not None and req_id in self._cancel_flags:
            self._cancel_flags[req_id] = True
            return True
        return False

    def request(self, payload: dict[str, Any], timeout: int) -> dict[str, Any] | None:
        if not ENABLE_PERSISTENT_WORKER or Process is None:
            return None
        if not self.is_alive():
            self.start()
        if not self.is_alive():
            return None
        try:
            # Correlate with an ID and route via round-robin to workers
            req_id = payload.get("id") or str(uuid.uuid4())
            payload = {**payload, "id": req_id}
            # Initialize cancellation flag in shared dict
            if self._cancel_flags is not None:
                self._cancel_flags[req_id] = False

            if not self.worker_conns:
                if self._cancel_flags and req_id in self._cancel_flags:
                    del self._cancel_flags[req_id]
                return None
                
            idx = self._next_idx % len(self.worker_conns)
            self._next_idx += 1
            conn = self.worker_conns[idx]
            
            # Serialize to JSON bytes (No Pickle!)
            try:
                # 1. Drain any stale data from previous timeouts
                while conn.poll(0):
                    try:
                        conn.recv_bytes()
                    except Exception:
                        pass

                msg_bytes = json.dumps(payload).encode("utf-8")
                conn.send_bytes(msg_bytes)
            except (OSError, ValueError, TypeError) as e:
                # Connection might be broken
                if self._cancel_flags and req_id in self._cancel_flags:
                    del self._cancel_flags[req_id]
                logger.warning(f"Failed to send request to worker: {e}")
                return None
                
            # Wait for response from THIS specific connection
            start_wait = time.time()
            while (time.time() - start_wait) < timeout:
                if conn.poll(0.1):
                    try:
                        resp_bytes = conn.recv_bytes()
                        resp = json.loads(resp_bytes.decode("utf-8"))
                        
                        # 2. Check Protocol ID
                        # If we received a stale message despite draining (race condition?), ignore it
                        if resp.get("id") == req_id:
                            if self._cancel_flags and req_id in self._cancel_flags:
                                del self._cancel_flags[req_id]
                            return resp
                    except Exception as e:
                        logger.warning(f"Error receiving/parsing worker response: {e}")
                        pass
            
            # Timeout loop finished
            if self._cancel_flags and req_id in self._cancel_flags:
                del self._cancel_flags[req_id]
                
            # Check if it was cancelled
            if self.cancel_request(req_id):
                 return {
                    "ok": False,
                    "error": "Request cancelled",
                    "error_code": "CANCELLED"
                }
                
            return None

        except (KeyboardInterrupt, SystemExit):
            # Stop workers and propagate interrupt to main process
            try:
                self.stop()
            except Exception:
                pass
            raise
        except (AttributeError, TypeError, ValueError, OSError) as e:
            # Specific exceptions that can occur in worker communication
            req_id = payload.get("id")
            if req_id and self._cancel_flags and req_id in self._cancel_flags:
                del self._cancel_flags[req_id]
            try:
                from .logging_config import safe_log

                safe_log(
                    "worker",
                    "warning",
                    f"Worker communication error, restarting: {e}",
                    exc_info=True,
                )
            except ImportError:
                pass
            try:
                self.stop()
                self.start()
                # Retry once logic omitted for simplicity/safety in this refactor
            except Exception:
                pass
            return None


def _worker_daemon_main(
    conn: Connection, stop_event: Any, cancel_flags: Any
) -> None:
    """Worker daemon main loop that processes requests from queue."""
    
    # Apply Unix resource limits (if applicable)
    try:
        _limit_resources()
    except (ImportError, OSError, ValueError, AttributeError):
        pass
        
    # SECURITY AUDIT: Removed WindowsJobObject usage from here.
    # Resource limits are now applied by the Supervisor (Parent Process)
    # via _WorkerManager.start(), avoiding ctypes usage in this worker.
    pass
        
    while True:
        if stop_event.is_set():
            break
        try:
            # Poll for input
            if conn.poll(0.1):
                msg_bytes = conn.recv_bytes()
                msg = json.loads(msg_bytes.decode("utf-8"))
            else:
                continue
        except (KeyboardInterrupt, SystemExit):
            # Don't re-raise in worker processes - just check stop_event and exit gracefully
            # The main process will handle KeyboardInterrupt and set stop_event
            if stop_event.is_set():
                break
            # If not stopped yet, set stop_event ourselves and exit
            try:
                stop_event.set()
            except (AttributeError, TypeError):
                pass
            break
        except Exception:
            # Connection error or EOF
            break
            
        try:
            kind = msg.get("type")
            req_id = msg.get("id")

            # Check cancellation before processing
            if cancel_flags and cancel_flags.get(req_id, False):
                resp = {
                        "ok": False,
                        "error": "Request cancelled",
                        "error_code": "CANCELLED",
                        "id": req_id,
                    }
                conn.send_bytes(json.dumps(resp, cls=SafeJSONEncoder).encode("utf-8"))
                continue

            if kind == "eval":
                pre = msg.get("preprocessed") or ""

                # Check for registry update
                registry_dump_json = msg.get("registry_dump_json")
                if registry_dump_json:
                    try:
                        from .function_manager import update_function_registry_from_dump

                        registry_dump = json.loads(registry_dump_json)
                        update_function_registry_from_dump(registry_dump)
                    except Exception:
                        # Log error but continue evaluation
                        pass

                # Extract allowed_functions from payload if present
                allowed_funcs_list = msg.get("allowed_functions")
                allowed_funcs = (
                    frozenset(allowed_funcs_list) if allowed_funcs_list else None
                )

                try:
                    # Get user-defined functions from the registry (populated via IPC)
                    from .function_manager import _function_registry
                    out = worker_evaluate(pre, allowed_functions=allowed_funcs, user_functions=_function_registry)
                except Exception as eval_error:
                    # Handle any errors in worker_evaluate gracefully
                    out = {
                        "ok": False,
                        "error": str(eval_error),
                        "error_code": "EVAL_ERROR",
                    }
                out["id"] = req_id
                # Check cancellation after processing
                if cancel_flags and cancel_flags.get(req_id, False):
                    out = {
                            "ok": False,
                            "error": "Request cancelled",
                            "error_code": "CANCELLED",
                            "id": req_id,
                        }
                
                conn.send_bytes(json.dumps(out, cls=SafeJSONEncoder).encode("utf-8"))
                
            elif kind == "solve":
                payload = msg.get("payload") or {}
                out = _worker_solve_dispatch(payload)
                out["id"] = req_id
                # Check cancellation after processing
                if cancel_flags and cancel_flags.get(req_id, False):
                    out = {
                            "ok": False,
                            "error": "Request cancelled",
                            "error_code": "CANCELLED",
                            "id": req_id,
                        }
                conn.send_bytes(json.dumps(out, cls=SafeJSONEncoder).encode("utf-8"))
            else:
                resp = {"ok": False, "error": "Unknown request type", "id": req_id}
                conn.send_bytes(json.dumps(resp, cls=SafeJSONEncoder).encode("utf-8"))
        except Exception as e:
            # Log the full error for debugging, but provide user-friendly message
            error_msg = str(e)
            # On Windows, resource module errors are expected - provide clearer message
            if "resource" in error_msg.lower() and "no module" in error_msg.lower():
                error_msg = (
                    "Resource limits unavailable on Windows (expected limitation)"
                )
            resp = {
                    "ok": False,
                    "error": f"Worker daemon error: {error_msg}",
                    "id": msg.get("id"),
                }
            try:
                conn.send_bytes(json.dumps(resp, cls=SafeJSONEncoder).encode("utf-8"))
            except Exception:
                pass
            




def _worker_solve_dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        eqs_input = payload.get("equations", [])
        eq_objs = []
        for item in eqs_input:
            lhs_s = item.get("lhs")
            rhs_s = item.get("rhs")
            if lhs_s is None or rhs_s is None:
                continue
            lhs_expr = parse_preprocessed(lhs_s)
            rhs_expr = parse_preprocessed(rhs_s)
            eq_objs.append(sp.Eq(lhs_expr, rhs_expr))
        if not eq_objs:
            return {
                "ok": False,
                "error": "No valid equations provided to worker-solve.",
            }
        try:
            _limit_resources()
        except (ImportError, OSError, ValueError, AttributeError):
            # Resource limits failed - log but continue
                try:
                    from .logging_config import safe_log

                    safe_log(
                        "worker",
                        "warning",
                        "Failed to apply resource limits",
                        exc_info=True,
                    )
                except ImportError:
                    pass
        solutions = sp.solve(eq_objs, dict=True)
        if not solutions:
            # Analyze why no solutions were found
            error_hints = []
            # Check for obviously impossible equations
            for eq in eq_objs:
                # Check for sin/cos/tan with impossible values
                if eq.has(sp.sin, sp.cos, sp.tan):
                    # Try to detect if any trig equation is impossible
                    if eq.has(sp.sin):
                        # Check if sin(x) = something where |something| > 1
                        try:
                            # Try to extract the value being compared
                            if eq.lhs.has(sp.sin) and not eq.rhs.has(sp.sin):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)"
                                    )
                            elif eq.rhs.has(sp.sin) and not eq.lhs.has(sp.sin):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)"
                                    )
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if eq.has(sp.cos):
                        try:
                            if eq.lhs.has(sp.cos) and not eq.rhs.has(sp.cos):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)"
                                    )
                            elif eq.rhs.has(sp.cos) and not eq.lhs.has(sp.cos):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)"
                                    )
                        except (ValueError, TypeError, AttributeError):
                            pass

            error_msg = "No solution found for this system of equations."
            if error_hints:
                error_msg += " Possible reasons:\n" + "\n".join(
                    f"  - {hint}" for hint in error_hints
                )
            else:
                error_msg += " The system may be inconsistent, overdetermined, or have no real solutions. Check for contradictory equations."
            return {
                "ok": False,
                "error": error_msg,
                "error_code": "NO_SOLUTION",
            }
        # Filter solutions to only include real ones
        # Import tolerance constant
        from .config import NUMERIC_TOLERANCE

        real_sols = []
        complex_sols = []
        for sol in solutions:
            is_real = True
            for _var, val in sol.items():
                try:
                    # Check if the value is real (imaginary part is negligible)
                    num_val = sp.N(val)
                    if abs(sp.im(num_val)) >= NUMERIC_TOLERANCE:
                        is_real = False
                        break
                except (ValueError, TypeError, AttributeError):
                    # If we can't evaluate, assume it might be complex
                    # Check if it's obviously complex (contains I or complex operations)
                    val_str = str(val)
                    if (
                        "I" in val_str
                        or "asin(" in val_str.lower()
                        or "acos(" in val_str.lower()
                    ):
                        # Check if asin/acos would produce complex (e.g., asin(pi) is complex)
                        if "asin" in val_str.lower():
                            try:
                                # Try to extract what's inside asin
                                import re

                                match = re.search(
                                    r"asin\(([^)]+)\)", val_str, re.IGNORECASE
                                )
                                if match:
                                    inner = match.group(1)
                                    inner_val = float(sp.N(inner))
                                    if abs(inner_val) > 1:
                                        is_real = False
                                        break
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if "acos" in val_str.lower():
                            try:
                                match = re.search(
                                    r"acos\(([^)]+)\)", val_str, re.IGNORECASE
                                )
                                if match:
                                    inner = match.group(1)
                                    inner_val = float(sp.N(inner))
                                    if abs(inner_val) > 1:
                                        is_real = False
                                        break
                            except (ValueError, TypeError, AttributeError):
                                pass
            if is_real:
                real_sols.append(sol)
            else:
                complex_sols.append(sol)

        # If we have only complex solutions, provide helpful error
        if not real_sols and complex_sols:
            error_hints = []
            for eq in eq_objs:
                if eq.has(sp.sin, sp.cos, sp.tan):
                    if eq.has(sp.sin):
                        try:
                            if eq.lhs.has(sp.sin) and not eq.rhs.has(sp.sin):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)"
                                    )
                            elif eq.rhs.has(sp.sin) and not eq.lhs.has(sp.sin):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)"
                                    )
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if eq.has(sp.cos):
                        try:
                            if eq.lhs.has(sp.cos) and not eq.rhs.has(sp.cos):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)"
                                    )
                            elif eq.rhs.has(sp.cos) and not eq.lhs.has(sp.cos):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(
                                        f"Equation '{eq}' has no real solutions: cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)"
                                    )
                        except (ValueError, TypeError, AttributeError):
                            pass
            error_msg = "No real solutions found for this system of equations (only complex solutions exist)."
            if error_hints:
                error_msg += " Reasons:\n" + "\n".join(
                    f"  - {hint}" for hint in error_hints
                )
            else:
                error_msg += (
                    " The system may have complex solutions but no real solutions."
                )
            return {
                "ok": False,
                "error": error_msg,
                "error_code": "NO_REAL_SOLUTIONS",
            }

        # Return only real solutions
        sols = []
        for sol in real_sols:
            sols.append({str(k): str(v) for k, v in sol.items()})
        return {"ok": True, "type": "system", "solutions": sols}
    except Exception as e:
        return {
            "ok": False,
            "error": f"Solver error: {e}",
            "error_code": "SOLVER_ERROR",
        }


_WORKER_MANAGER = _WorkerManager()


def warmup_workers() -> None:
    """Pre-initialize worker processes to avoid startup delay on first calculation.

    This function starts worker processes early so that the first calculation
    doesn't have to wait for process spawning and module imports.
    """
    if ENABLE_PERSISTENT_WORKER and Process is not None:
        try:
            if not _WORKER_MANAGER.is_alive():
                _WORKER_MANAGER.start()
                # Send a warmup request to ensure workers are fully initialized
                # This triggers module imports in worker processes
                try:
                    _WORKER_MANAGER.request(
                        {"type": "eval", "preprocessed": "1"}, timeout=2
                    )
                except Exception:
                    # Ignore warmup errors - workers will be ready on real request
                    pass
        except Exception:
            # If warmup fails, workers will start on first real request
            pass


def _worker_eval_cached(
    preprocessed_expr: str,
    context_hash: str | None = None,
    registry_dump_json: str | None = None,
    allowed_functions: frozenset[str] | None = None,
    use_cache: bool = True,
) -> str:
    """Evaluate expression with persistent cache support."""
    # Check persistent cache first
    try:
        from .cache_manager import get_cache_hits
        from .cache_manager import get_cached_eval

        if use_cache:
            cached_result = get_cached_eval(preprocessed_expr, context_hash)
            if cached_result is not None:
                if logger:
                    logger.debug(f"Cache hit for: {preprocessed_expr[:50]}")
                # Cache hit was tracked by get_cached_eval above
                # Get the cache hits from this process and attach them to the result
                worker_cache_hits = get_cache_hits()
                # Always attach cache hits - if get_cache_hits() didn't return it, add it manually
                if not worker_cache_hits:
                    # Manual fallback: add the current expression as a cache hit
                    worker_cache_hits = [(preprocessed_expr, "eval")]
                try:
                    cached_data = json.loads(cached_result)
                    # Always add cache hits
                    cached_data["cache_hits"] = worker_cache_hits
                    return json.dumps(cached_data)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, return original (old format)
                    # Create new dict with cache hit info
                    try:
                        return json.dumps(
                            {
                                "ok": True,
                                "result": (
                                    cached_result
                                    if isinstance(cached_result, str)
                                    else json.loads(cached_result).get("result", "")
                                ),
                                "cache_hits": (
                                    worker_cache_hits
                                    if worker_cache_hits
                                    else [(preprocessed_expr, "eval")]
                                ),
                            }
                        )
                    except Exception:
                        pass
                return cached_result
    except ImportError:
        pass

    # Not in persistent cache, evaluate normally and measure time
    start_time = time.perf_counter()
    resp = _WORKER_MANAGER.request(
        {
            "type": "eval",
            "preprocessed": preprocessed_expr,
            "registry_dump_json": registry_dump_json,
            "allowed_functions": list(allowed_functions) if allowed_functions else None,
        },
        timeout=WORKER_TIMEOUT,
    )
    compute_time = time.perf_counter() - start_time

    if isinstance(resp, dict):
        result_json = json.dumps(resp)
        # Save to persistent cache if evaluation was successful
        try:
            from .cache_manager import update_eval_cache
            from .cache_manager import update_subexpr_cache

            if resp.get("ok") and use_cache:
                update_eval_cache(
                    preprocessed_expr, result_json, compute_time, context_hash
                )
                # Also cache as sub-expression if it's a simple numeric result
                result_value = str(resp.get("result", ""))
                approx_value = str(resp.get("approx", ""))
                # Only cache pure numeric expressions (no variables)
                if result_value and not any(
                    c in result_value
                    for c in ["x", "y", "z", "X", "Y", "Z", "a", "b", "c"]
                ):
                    # Cache the sub-expression mapping
                    cache_value = approx_value if approx_value else result_value
                    if cache_value:
                        update_subexpr_cache(
                            preprocessed_expr, cache_value, compute_time
                        )
        except ImportError:
            pass
        return result_json
    cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
    try:
        start_time_subproc = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 bytes instead of raising error
        )
        compute_time = time.perf_counter() - start_time_subproc
        result_text = proc.stdout or ""
        # Try to save to persistent cache
        try:
            from .cache_manager import update_eval_cache
            from .cache_manager import update_subexpr_cache

            try:
                result_data = json.loads(result_text)
                if result_data.get("ok"):
                    update_eval_cache(
                        preprocessed_expr, result_text, compute_time, context_hash
                    )
                    result_value = result_data.get("result", "")
                    approx_value = result_data.get("approx", "")
                    # Only cache pure numeric expressions
                    if result_value and not any(
                        c in result_value
                        for c in ["x", "y", "z", "X", "Y", "Z", "a", "b", "c"]
                    ):
                        cache_value = approx_value if approx_value else result_value
                        if cache_value:
                            update_subexpr_cache(
                                preprocessed_expr, cache_value, compute_time
                            )
            except (json.JSONDecodeError, KeyError):
                pass
        except ImportError:
            pass
        return result_text
    except UnicodeDecodeError:
        # Fallback if UTF-8 decoding fails
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=WORKER_TIMEOUT,
        )
        return proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""


@lru_cache(maxsize=CACHE_SIZE_SOLVE)
def _worker_solve_cached(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Invalid JSON - log and use empty equations
        try:
            from .logging_config import safe_log

            safe_log("worker", "warning", f"Invalid JSON payload in solve cache: {e}")
        except ImportError:
            pass
        payload = {"equations": []}
    resp = _WORKER_MANAGER.request(
        {"type": "solve", "payload": payload}, timeout=WORKER_TIMEOUT
    )
    if isinstance(resp, dict):
        return json.dumps(resp)
    cmd = _build_self_cmd(["--worker-solve", "--payload", payload_json])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 bytes instead of raising error
        )
        return proc.stdout or ""
    except UnicodeDecodeError:
        # Fallback if UTF-8 decoding fails
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=WORKER_TIMEOUT,
        )
        return proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""


def evaluate_safely(
    expr: str,
    timeout: int = WORKER_TIMEOUT,
    allowed_functions: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Safely evaluate an expression string via worker sandbox."""

    from .cache_manager import clear_cache_hits
    from .cache_manager import get_cache_hits
    from .parser import preprocess

    # Clear cache hits at the start (before any operations)
    clear_cache_hits()

    # Track sub-expression cache hits from preprocessing
    subexpr_cache_hits: list[tuple[str, str]] = []
    try:
        from .function_manager import get_function_registry_dump
        from .function_manager import get_function_registry_hash

        # Try to get registry hash/dump, but handle missing context (legacy callers)
        try:
            context_hash = get_function_registry_hash() # type: ignore
            registry_dump = get_function_registry_dump() # type: ignore
            registry_dump_json = json.dumps(registry_dump)
        except (TypeError, NameError, AttributeError):
            # If get_function_registry_hash requires context but we don't have it
            context_hash = ""
            registry_dump_json = None

        pre = preprocess(expr, allowed_functions=allowed_functions)

        # Capture sub-expression cache hits from preprocessing (in main process)
        subexpr_cache_hits = get_cache_hits()
    except ValidationError as e:
        return {"ok": False, "error": str(e), "error_code": e.code}
    except ValueError as e:
        return {
            "ok": False,
            "error": f"Preprocess error: {e}",
            "error_code": "PREPROCESS_ERROR",
        }
    except (TypeError, AttributeError) as e:
        # Unexpected error in preprocessing - log it
        try:
            from .logging_config import safe_log

            safe_log(
                "worker", "error", f"Unexpected preprocessing error: {e}", exc_info=True
            )
        except ImportError:
            pass
        return {"ok": False, "error": "Preprocess error", "error_code": "UNKNOWN_ERROR"}
    try:
        try:
            # Check if any user function OR unknown function is used in the expression
            # If so, we bypass the cache logic to prevent stale cache issues
            use_cache = True
            
            # Use regex to find all function calls "name("
            import re
            # Find all words followed by open paren
            # We ignore words starting with digits to avoid 2(x) issues (though 2( isn't valid func syntax)
            
            # Lazy load ALLOWED_SYMPY_NAMES if needed
            from .config import ALLOWED_SYMPY_NAMES
            
            # Pattern: matches "func(" where func is a valid identifier
            potential_funcs = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", pre)
            
            for func_name in potential_funcs:
                # If we find a function that is NOT in the allowed SymPy list,
                # it is either user-defined or unknown (which might become defined).
                # In either case, we should NOT CACHE it.
                if func_name not in ALLOWED_SYMPY_NAMES:
                    use_cache = False
                    break
        except Exception:
            use_cache = True # Fallback to caching if check fails

        stdout_text = _worker_eval_cached(
            pre, context_hash, registry_dump_json, allowed_functions=allowed_functions, use_cache=use_cache
        )
        # Cache hits are now embedded in the JSON response from _worker_eval_cached
        # So we'll extract them after parsing JSON below
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Evaluation timed out.", "error_code": "TIMEOUT"}
    except (OSError, ValueError) as e:
        # Specific exceptions for worker communication failures
        try:
            from .logging_config import safe_log

            safe_log(
                "worker", "error", f"Worker communication error: {e}", exc_info=True
            )
        except ImportError:
            pass
        return {
            "ok": False,
            "error": "Worker communication failed",
            "error_code": "COMM_ERROR",
        }
    try:
        data = json.loads(stdout_text)
        # Cache hits are now embedded in the JSON from _worker_eval_cached
        # Extract them if present, otherwise ensure it's an empty list
        worker_hits = data.get("cache_hits", [])
        # JSON deserializes tuples as lists, so convert back to tuples for consistency
        worker_hits_tuples = [
            tuple(hit) if isinstance(hit, list) else hit for hit in worker_hits
        ]
        # Merge sub-expression cache hits (from preprocessing) with worker cache hits
        # Combine both lists (avoid duplicates)
        combined_hits = list(worker_hits_tuples)
        for hit in subexpr_cache_hits:
            # Convert to tuple if needed
            hit_tuple = tuple(hit) if not isinstance(hit, tuple) else hit
            if hit_tuple not in combined_hits:
                combined_hits.append(hit_tuple)
        # Always set cache_hits (even if empty) for consistency
        data["cache_hits"] = combined_hits
        return data
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Specific exceptions for JSON parsing
        return {
            "ok": False,
            "error": f"Invalid worker output: {e}.",
            "error_code": "INVALID_OUTPUT",
        }
    except Exception as e:
        # Catch-all for truly unexpected errors - log full traceback
        logger.exception("Unexpected error parsing worker output")
        return {
            "ok": False,
            "error": f"Invalid worker output: {e}",
            "error_code": "UNKNOWN_ERROR",
        }


def clear_caches() -> None:
    """Clear worker-side LRU caches, parser cache, and persistent cache."""
    try:
        # Clear in-memory LRU caches
        if hasattr(_worker_eval_cached, "cache_clear"):
            _worker_eval_cached.cache_clear()  # Function may not be decorated with lru_cache anymore
        _worker_solve_cached.cache_clear()
        from .parser import parse_preprocessed as _pp

        _pp.cache_clear()

        # Clear persistent cache
        try:
            from .cache_manager import clear_persistent_cache

            clear_persistent_cache()
        except ImportError:
            pass
    except (ValueError, TypeError, AttributeError):
        # Expected for some cache operations
        pass


def cancel_current_request(req_id: str | None = None) -> bool:
    """Cancel a pending worker request. If req_id is None, attempts to cancel the most recent."""
    if req_id:
        return _WORKER_MANAGER.cancel_request(req_id)
    return False  # For now, requires explicit ID - can be enhanced


class Worker:
    def __init__(self, parser, solver, cache_manager, function_manager):
        self.parser = parser
        self.solver = solver
        self.cache_manager = cache_manager
        self.function_manager = function_manager
        self.variables = {}

    def process_command(self, command):
        """
        Process a single command string: check cache, parse, evaluate, store cache.
        """
        # 1. Get current context hash (state of functions)
        # Ensure we get the LATEST hash.
        context_hash = self.function_manager.get_registry_hash()

        # 2. Check Cache
        cached_result = self.cache_manager.get(command, context_hash)
        if cached_result is not None:
            return cached_result

        try:
            # 3. Parse
            tree = self.parser.parse(command)

            # 4. Evaluate
            result = self.evaluate_ast(tree)

            # 5. Cache Result
            # Note: If the command was an assignment, the hash has CHANGED inside evaluate_ast.
            # We should probably not cache 'Assignment' success messages under the OLD hash
            # if we want strict correctness, but it usually doesn't matter for strings.
            # However, for the NEXT command to work, the hash must be updated.

            self.cache_manager.set(command, context_hash, result)
            return result

        except Exception as e:
            return f"Error: {str(e)}"

    def evaluate_ast(self, node):
        """Recursively evaluate the AST."""
        node_type = node["type"]

        if node_type == "Assignment":
            # Extract the target variable name and value from the node
            var_name = node.get("name") or node.get("target")
            value_node = node.get("value")

            # Helper to check if it's a function definition (simplified)
            # Assuming parser marks function definitions clearly or we detect args
            is_func_def = "args" in node and node["args"] is not None

            if is_func_def:
                # It's a function definition: f(x) = ...
                func_args = node["args"]

                # We store the raw string expression or the AST for the function body
                # Assuming function_manager takes (name, args, body_node/str)
                # For this fix, we rely on existing logic, just triggering the update.

                # Note: The actual AST structure for assignment might differ slightly
                # depending on your parser. Assuming standard keys here.

                self.function_manager.add_function(var_name, func_args, value_node)

                # --- FIX FOR STALE CACHE ---
                # Force a hash regeneration/update immediately after definition
                # This ensures the NEXT command (like f(2)) sees the new hash.
                if hasattr(self.function_manager, "update_registry_hash"):
                    self.function_manager.update_registry_hash()
                elif hasattr(self.function_manager, "_registry_hash"):
                    # forceful invalidation if using cached_property
                    self.function_manager._registry_hash = None

                return f"Function '{var_name}' defined."

            else:
                # Variable assignment: a = 2
                value = self.evaluate_ast(value_node)
                self.variables[var_name] = value
                return f"{var_name} = {value}"

        elif node_type == "FunctionCall":
            # Existing logic
            func_name = node["name"]
            args = [self.evaluate_ast(arg) for arg in node["args"]]
            return self.solver.evaluate_function(func_name, args)

        elif node_type == "BinOp":
            left = self.evaluate_ast(node["left"])
            right = self.evaluate_ast(node["right"])
            op = node["op"]
            # This part needs actual SymPy evaluation based on the operator
            # For simplicity, returning a string representation
            return f"({left} {op} {right})"

        elif node_type == "Number":
            return float(node["value"])

        elif node_type == "Variable":
            var_name = node["name"]
            if var_name in self.variables:
                return self.variables[var_name]
            else:
                return f"Unresolved variable: {var_name}"

        elif node_type == "FunctionDefinition":
            # This case might be handled by the 'Assignment' block if it captures function definitions.
            # If your parser creates a distinct 'FunctionDefinition' node, handle it here.
            func_name = node["name"]
            func_args = node["args"]
            func_body = node["body"]

            self.function_manager.add_function(func_name, func_args, func_body)
            # --- FIX FOR STALE CACHE ---
            if hasattr(self.function_manager, "update_registry_hash"):
                self.function_manager.update_registry_hash()
            elif hasattr(self.function_manager, "_registry_hash"):
                self.function_manager._registry_hash = None
            return f"Function '{func_name}' defined."

        else:
            # Handle other node types or raise an error for unsupported ones
            return f"Unsupported node type: {node_type}"


# Alias for backward compatibility (used by CLI and other modules)
# evaluate_safely = worker_evaluate # REMOVED: This overwrites the actual evaluate_safely function defined above!
