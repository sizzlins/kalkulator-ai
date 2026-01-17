"""Numba-accelerated RPN evaluator for ExpressionTree.

This module provides a JIT-compiled interpreter that is ~10x faster than
the pure Python RPN evaluator for large datasets.

Falls back to pure Python if Numba is not installed.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create a no-op decorator for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# Opcode definitions (must match UNARY_OPCODES and BINARY_OPCODES)
# Unary opcodes: 0-99
OP_SIN = 0
OP_COS = 1
OP_TAN = 2
OP_EXP = 3
OP_LOG = 4
OP_SQRT = 5
OP_ABS = 6
OP_NEG = 7
OP_INV = 8
OP_SQUARE = 9
OP_CUBE = 10
OP_SINH = 11
OP_COSH = 12
OP_TANH = 13
OP_FLOOR = 14
OP_CEIL = 15
OP_SIGN = 16
OP_ROUND = 17
OP_ATAN = 18
OP_ASIN = 19
OP_ACOS = 20

# Binary opcodes: 100-199
OP_ADD = 100
OP_SUB = 101
OP_MUL = 102
OP_DIV = 103
OP_POW = 104
OP_MAX = 105
OP_MIN = 106
OP_MOD = 107
OP_ATAN2 = 108

# Special opcodes
OP_CONST = 200  # Push constant
OP_VAR = 201    # Push variable column

# Mapping from operator names to opcodes
UNARY_OPCODES = {
    "sin": OP_SIN, "cos": OP_COS, "tan": OP_TAN, "exp": OP_EXP, "log": OP_LOG,
    "plog": OP_LOG, "sqrt": OP_SQRT, "psqrt": OP_SQRT, "abs": OP_ABS,
    "neg": OP_NEG, "inv": OP_INV, "square": OP_SQUARE, "cube": OP_CUBE,
    "sinh": OP_SINH, "cosh": OP_COSH, "tanh": OP_TANH, "floor": OP_FLOOR,
    "ceil": OP_CEIL, "ceiling": OP_CEIL, "sign": OP_SIGN, "round": OP_ROUND,
    "atan": OP_ATAN, "asin": OP_ASIN, "acos": OP_ACOS,
}

BINARY_OPCODES = {
    "add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV, "pow": OP_POW,
    "max": OP_MAX, "min": OP_MIN, "mod": OP_MOD, "atan2": OP_ATAN2,
}


def compile_rpn_numba(raw_tokens: list, var_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compile RPN tokens to Numba-compatible arrays.
    
    Returns:
        (opcodes, values) tuple where:
        - opcodes: int32 array of operation codes
        - values: float64 array of constants/var indices
    """
    opcodes = []
    values = []
    
    for type_code, val in raw_tokens:
        if type_code == 'CONST':
            opcodes.append(OP_CONST)
            values.append(float(val))
        elif type_code == 'VAR':
            opcodes.append(OP_VAR)
            values.append(float(var_map.get(val, 0)))
        elif type_code == 'UNARY':
            opcode = UNARY_OPCODES.get(val, -1)
            if opcode >= 0:
                opcodes.append(opcode)
                values.append(0.0)  # Placeholder
        elif type_code == 'BINARY':
            opcode = BINARY_OPCODES.get(val, -1)
            if opcode >= 0:
                opcodes.append(opcode)
                values.append(0.0)  # Placeholder
    
    return np.array(opcodes, dtype=np.int32), np.array(values, dtype=np.float64)


@njit(cache=True)
def _safe_div(a, b):
    """Safe division with epsilon."""
    return a / (b + 1e-10 * np.sign(b + 1e-300))


@njit(cache=True)
def _safe_log(x):
    """Safe log with epsilon."""
    return np.log(np.abs(x) + 1e-10)


@njit(cache=True)
def _safe_sqrt(x):
    """Safe sqrt."""
    return np.sqrt(np.abs(x))


@njit(cache=True)
def _safe_pow(x, y):
    """Safe power with clipping."""
    y_clip = np.clip(y, -100, 100)
    x_clip = np.clip(x, -1e150, 1e150)
    # Use abs for negative bases
    result = np.power(np.abs(x_clip) + 1e-300, y_clip)
    return np.clip(result, -1e100, 1e100)


@njit(cache=True, parallel=False, fastmath=True)
def evaluate_rpn_numba(opcodes: np.ndarray, values: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Numba-JIT compiled RPN evaluator.
    
    Args:
        opcodes: int32 array of operation codes
        values: float64 array of constants/variable indices
        X: (n_samples, n_vars) input array
        
    Returns:
        (n_samples,) result array
    """
    n_samples = X.shape[0]
    n_ops = len(opcodes)
    
    # Fixed-size stack (max depth 64 should be enough)
    MAX_STACK = 64
    stack = np.zeros((MAX_STACK,), dtype=np.float64)
    sp = 0  # Stack pointer
    
    # Result array
    result = np.zeros(n_samples, dtype=np.float64)
    
    # Process each sample
    for i in range(n_samples):
        sp = 0  # Reset stack
        
        for j in range(n_ops):
            op = opcodes[j]
            val = values[j]
            
            # Constants and Variables
            if op == OP_CONST:
                stack[sp] = val
                sp += 1
            elif op == OP_VAR:
                var_idx = int(val)
                stack[sp] = X[i, var_idx]
                sp += 1
            
            # Unary operations
            elif op == OP_SIN:
                stack[sp-1] = np.sin(stack[sp-1])
            elif op == OP_COS:
                stack[sp-1] = np.cos(stack[sp-1])
            elif op == OP_TAN:
                stack[sp-1] = np.tan(stack[sp-1])
            elif op == OP_EXP:
                stack[sp-1] = np.exp(np.clip(stack[sp-1], -700, 700))
            elif op == OP_LOG:
                stack[sp-1] = np.log(np.abs(stack[sp-1]) + 1e-10)
            elif op == OP_SQRT:
                stack[sp-1] = np.sqrt(np.abs(stack[sp-1]))
            elif op == OP_ABS:
                stack[sp-1] = np.abs(stack[sp-1])
            elif op == OP_NEG:
                stack[sp-1] = -stack[sp-1]
            elif op == OP_INV:
                stack[sp-1] = 1.0 / (stack[sp-1] + 1e-10)
            elif op == OP_SQUARE:
                stack[sp-1] = stack[sp-1] * stack[sp-1]
            elif op == OP_CUBE:
                stack[sp-1] = stack[sp-1] * stack[sp-1] * stack[sp-1]
            elif op == OP_SINH:
                stack[sp-1] = np.sinh(np.clip(stack[sp-1], -700, 700))
            elif op == OP_COSH:
                stack[sp-1] = np.cosh(np.clip(stack[sp-1], -700, 700))
            elif op == OP_TANH:
                stack[sp-1] = np.tanh(stack[sp-1])
            elif op == OP_FLOOR:
                stack[sp-1] = np.floor(stack[sp-1])
            elif op == OP_CEIL:
                stack[sp-1] = np.ceil(stack[sp-1])
            elif op == OP_SIGN:
                v = stack[sp-1]
                if v > 0: stack[sp-1] = 1.0
                elif v < 0: stack[sp-1] = -1.0
                else: stack[sp-1] = 0.0
            elif op == OP_ROUND:
                stack[sp-1] = np.round(stack[sp-1])
            elif op == OP_ATAN:
                stack[sp-1] = np.arctan(stack[sp-1])
            elif op == OP_ASIN:
                stack[sp-1] = np.arcsin(np.clip(stack[sp-1], -1, 1))
            elif op == OP_ACOS:
                stack[sp-1] = np.arccos(np.clip(stack[sp-1], -1, 1))
            
            # Binary operations
            elif op == OP_ADD:
                sp -= 1
                stack[sp-1] = stack[sp-1] + stack[sp]
            elif op == OP_SUB:
                sp -= 1
                stack[sp-1] = stack[sp-1] - stack[sp]
            elif op == OP_MUL:
                sp -= 1
                stack[sp-1] = stack[sp-1] * stack[sp]
            elif op == OP_DIV:
                sp -= 1
                stack[sp-1] = stack[sp-1] / (stack[sp] + 1e-10)
            elif op == OP_POW:
                sp -= 1
                base = stack[sp-1]
                exp = np.clip(stack[sp], -100, 100)
                stack[sp-1] = np.power(np.abs(base) + 1e-300, exp)
            elif op == OP_MAX:
                sp -= 1
                if stack[sp-1] > stack[sp]:
                    pass  # Keep left
                else:
                    stack[sp-1] = stack[sp]
            elif op == OP_MIN:
                sp -= 1
                if stack[sp-1] < stack[sp]:
                    pass  # Keep left
                else:
                    stack[sp-1] = stack[sp]
            elif op == OP_MOD:
                sp -= 1
                stack[sp-1] = np.fmod(stack[sp-1], stack[sp] + 1e-10)
            elif op == OP_ATAN2:
                sp -= 1
                stack[sp-1] = np.arctan2(stack[sp-1], stack[sp])
        
        # Final result
        if sp > 0:
            result[i] = stack[0]
        else:
            result[i] = 0.0
    
    return result


def evaluate_rpn_fast(opcodes: np.ndarray, values: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Fast RPN evaluator with Numba fallback.
    
    Uses Numba-JIT if available, otherwise falls back to pure Python.
    """
    if HAS_NUMBA:
        try:
            return evaluate_rpn_numba(opcodes, values, X)
        except Exception:
            pass
    
    # Fallback: pure Python vectorized (current implementation)
    # This is called if Numba fails or is not installed
    return _evaluate_rpn_python(opcodes, values, X)


def _evaluate_rpn_python(opcodes: np.ndarray, values: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pure Python vectorized RPN evaluator (fallback)."""
    n_samples = X.shape[0]
    stack = []
    
    for j in range(len(opcodes)):
        op = opcodes[j]
        val = values[j]
        
        if op == OP_CONST:
            stack.append(np.full(n_samples, val))
        elif op == OP_VAR:
            stack.append(X[:, int(val)])
        # Unary ops
        elif op == OP_SIN:
            stack[-1] = np.sin(stack[-1])
        elif op == OP_COS:
            stack[-1] = np.cos(stack[-1])
        elif op == OP_TAN:
            stack[-1] = np.tan(stack[-1])
        elif op == OP_EXP:
            stack[-1] = np.exp(np.clip(stack[-1], -700, 700))
        elif op == OP_LOG:
            stack[-1] = np.log(np.abs(stack[-1]) + 1e-10)
        elif op == OP_SQRT:
            stack[-1] = np.sqrt(np.abs(stack[-1]))
        elif op == OP_ABS:
            stack[-1] = np.abs(stack[-1])
        elif op == OP_NEG:
            stack[-1] = -stack[-1]
        elif op == OP_INV:
            stack[-1] = 1.0 / (stack[-1] + 1e-10)
        elif op == OP_SQUARE:
            stack[-1] = stack[-1] ** 2
        elif op == OP_CUBE:
            stack[-1] = stack[-1] ** 3
        elif op == OP_SINH:
            stack[-1] = np.sinh(np.clip(stack[-1], -700, 700))
        elif op == OP_COSH:
            stack[-1] = np.cosh(np.clip(stack[-1], -700, 700))
        elif op == OP_TANH:
            stack[-1] = np.tanh(stack[-1])
        elif op == OP_FLOOR:
            stack[-1] = np.floor(stack[-1])
        elif op == OP_CEIL:
            stack[-1] = np.ceil(stack[-1])
        elif op == OP_SIGN:
            stack[-1] = np.sign(stack[-1])
        elif op == OP_ROUND:
            stack[-1] = np.round(stack[-1])
        elif op == OP_ATAN:
            stack[-1] = np.arctan(stack[-1])
        elif op == OP_ASIN:
            stack[-1] = np.arcsin(np.clip(stack[-1], -1, 1))
        elif op == OP_ACOS:
            stack[-1] = np.arccos(np.clip(stack[-1], -1, 1))
        # Binary ops
        elif op == OP_ADD:
            r = stack.pop()
            stack[-1] = stack[-1] + r
        elif op == OP_SUB:
            r = stack.pop()
            stack[-1] = stack[-1] - r
        elif op == OP_MUL:
            r = stack.pop()
            stack[-1] = stack[-1] * r
        elif op == OP_DIV:
            r = stack.pop()
            stack[-1] = stack[-1] / (r + 1e-10)
        elif op == OP_POW:
            r = stack.pop()
            stack[-1] = np.power(np.abs(stack[-1]) + 1e-300, np.clip(r, -100, 100))
        elif op == OP_MAX:
            r = stack.pop()
            stack[-1] = np.maximum(stack[-1], r)
        elif op == OP_MIN:
            r = stack.pop()
            stack[-1] = np.minimum(stack[-1], r)
        elif op == OP_MOD:
            r = stack.pop()
            stack[-1] = np.fmod(stack[-1], r + 1e-10)
        elif op == OP_ATAN2:
            r = stack.pop()
            stack[-1] = np.arctan2(stack[-1], r)
    
    if stack:
        result = stack[0]
        if np.isscalar(result):
            return np.full(n_samples, result)
        return result
    return np.zeros(n_samples)
