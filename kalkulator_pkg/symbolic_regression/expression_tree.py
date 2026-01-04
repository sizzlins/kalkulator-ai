"""Expression Tree data structure for Genetic Programming Symbolic Regression.

This module implements the core data structure for representing mathematical
expressions as trees that can be evolved through genetic programming.

Key Classes:
    - NodeType: Enum for terminal/operator node types
    - ExpressionNode: Single node in the expression tree
    - ExpressionTree: Complete tree with evaluation and manipulation methods
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from typing import Any
from typing import Callable

import numpy as np
import sympy as sp
from scipy import special as scipy_special  # For Bessel functions


class NodeType(Enum):
    """Types of nodes in an expression tree."""

    CONSTANT = auto()  # Numeric constant (e.g., 3.14)
    VARIABLE = auto()  # Input variable (e.g., x, y, z)
    UNARY_OP = auto()  # Unary operator (e.g., sin, cos, exp)
    BINARY_OP = auto()  # Binary operator (e.g., +, -, *, /)


# Operator definitions with arities and safe evaluation functions
def safe_tan(x):
    return np.tan(np.clip(x, -1e6, 1e6))


def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))


def safe_log(x):
    # Use scimath.log (handles negative inputs -> complex)
    # Add epsilon to magnitude to avoid log(0)
    # x + epsilon is tricky for complex, so we ensure x isn't exactly 0
    safe_x = np.where(x == 0, 1e-10, x)
    return np.lib.scimath.log(safe_x)


def safe_sqrt(x):
    # Use scimath.sqrt to handle negative inputs
    return np.lib.scimath.sqrt(x)


def safe_inv(x):
    # Handle complex division near zero
    # Add small epsilon in direction of x (complex-aware)
    safe_x = x + 1e-10 * (x / (np.abs(x) + 1e-10))
    return 1.0 / safe_x


def safe_mul(x, y):
    return np.clip(x * y, -1e100, 1e100)


def safe_div(x, y):
    safe_y = y + 1e-10 * (y / (np.abs(y) + 1e-10))
    return x / safe_y


def safe_pow(x, y):
    """Safe power operation with reasonable limits but supporting x^x features.
    
    Supersedes the old aggressive clipping version.
    Now handles negative bases by returning complex numbers when needed.
    """
    # Clip exponent to reasonable range that supports x^x up to ~100
    # 100^100 is 1e200, which fits in float64.
    y_clipped = np.clip(y, -100, 100)
    
    # Clip base magnitude to prevent immediate overflow on large bases
    # e.g. 1e150^2 -> 1e300.
    x_clipped = np.clip(x, -1e150, 1e150)
    
    try:
        # Determine if we need complex arithmetic
        # Trigger if inputs are complex OR if base < 0 and exponent is fractional
        is_complex = np.iscomplexobj(x) or np.iscomplexobj(y)
        
        if not is_complex:
            # Check for negative bases combined with fractional exponents
            # This would produce NaNs in real arithmetic
            has_neg_base = np.any(x_clipped < 0)
            if has_neg_base:
                # Check if any corresponding exponents are non-integers
                # We use a relaxed tolerance for "integer-ness"
                is_integer_exp = np.abs(y_clipped - np.round(y_clipped)) < 1e-9
                
                # If we have negative base AND non-integer exponent -> need complex
                # We can do this element-wise or just switch to complex mode globally
                # Global switch is safer and simpler
                if np.any((x_clipped < 0) & (~is_integer_exp)):
                    is_complex = True

        if is_complex:
            # Cast to complex to allow negative base powers
            # e.g. (-2.5)^(-2.5) -> complex
            x_complex = x_clipped.astype(np.complex128)
            y_complex = y_clipped.astype(np.complex128)
            result = np.power(x_complex, y_complex)
            
            # Clip magnitude to prevent overflow
            mag = np.abs(result)
            mask = mag > 1e100
            if np.any(mask):
                 # Scale down to limit while preserving phase
                 scale = 1e100 / (mag[mask] + 1e-300)
                 result[mask] *= scale
            return result
        else:
            # Pure real path
            result = np.power(x_clipped, y_clipped)
            return np.clip(result, -1e100, 1e100)
            
    except Exception as e:
        # print(f"DEBUG: safe_pow crashed: {e}") 
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0



def safe_sinh(x):
    return np.sinh(np.clip(x, -700, 700))


def safe_cosh(x):
    return np.cosh(np.clip(x, -700, 700))


def safe_square(x):
    return np.clip(x * x, -1e100, 1e100)


# Protected Operators (Agent Handoff Rule 5: Root Cause)
# These prevent complex/NaN values and "max()" safety patches
# that trap the optimizer.

def psqrt(x):
    """Protected Sqrt: sqrt(abs(x)). Returns real float."""
    return np.sqrt(np.abs(x))


def plog(x):
    """Protected Log: log(abs(x) + epsilon). Returns real float."""
    return np.log(np.abs(x) + 1e-10)



# Protected Operators (Agent Handoff Rule 5: Root Cause)
# These prevent complex/NaN values and "max()" safety patches
# that trap the optimizer.

def psqrt(x):
    """Protected Sqrt: sqrt(abs(x)). Returns real float."""
    return np.sqrt(np.abs(x))


def plog(x):
    """Protected Log: log(abs(x) + epsilon). Returns real float."""
    return np.log(np.abs(x) + 1e-10)



def safe_cube(x):
    return np.clip(x * x * x, -1e100, 1e100)


def safe_prime_pi(x):
    """Count primes <= x. Vectorized for numpy arrays."""
    from sympy import primepi
    
    def _count_primes(val):
        # 1. Reject Complex / non-real
        if np.iscomplexobj(val):
            return 0.0
        if isinstance(val, complex):
            return 0.0
            
        # 2. Reject non-finite or negative
        if not np.isfinite(val) or val < 0:
            return 0.0
            
        # 3. Safeguard against large inputs (O(N!) prevention)
        # primepi gets slow for N > 10^9. Genetic engine calls it frequently.
        # Cap at 10^7 (10 million) to keep it responsive (approx 0.1ms)
        if val > 1_000_000:
             # Fast approximation for large x: x / ln(x)
             return float(val / np.log(val))

        try:
            return float(primepi(int(val)))
        except Exception:
            return 0.0
    
    return np.vectorize(_count_primes, otypes=[float])(x)


def safe_bitwise_xor(x, y):
    """Bitwise XOR for floating point inputs (casts to int)."""
    if not (np.isscalar(x) and np.isscalar(y)) or not (np.isreal(x) and np.isreal(y)):
        return 0.0
    # Reject complex/nan/inf
    if not (np.isfinite(x) and np.isfinite(y)):
        return 0.0
    try:
        return float(int(np.real(x)) ^ int(np.real(y)))
    except:
        return 0.0

def safe_bitwise_and(x, y):
    if not (np.isscalar(x) and np.isscalar(y)) or not (np.isreal(x) and np.isreal(y)): return 0.0
    if not (np.isfinite(x) and np.isfinite(y)): return 0.0
    try: return float(int(np.real(x)) & int(np.real(y)))
    except: return 0.0

def safe_bitwise_or(x, y):
    if not (np.isscalar(x) and np.isscalar(y)) or not (np.isreal(x) and np.isreal(y)): return 0.0
    if not (np.isfinite(x) and np.isfinite(y)): return 0.0
    try: return float(int(np.real(x)) | int(np.real(y)))
    except: return 0.0

def safe_lshift(x, y):
    if not (np.isscalar(x) and np.isscalar(y)) or not (np.isreal(x) and np.isreal(y)): return 0.0
    if not (np.isfinite(x) and np.isfinite(y)): return 0.0
    try:
        iy = int(np.real(y))
        if iy < 0 or iy > 64: return 0.0 # Cap shift
        return float(int(np.real(x)) << iy)
    except: return 0.0

def safe_rshift(x, y):
    if not (np.isscalar(x) and np.isscalar(y)) or not (np.isreal(x) and np.isreal(y)): return 0.0
    if not (np.isfinite(x) and np.isfinite(y)): return 0.0
    try:
        iy = int(np.real(y))
        if iy < 0 or iy > 64: return 0.0
        return float(int(np.real(x)) >> iy)
    except: return 0.0





UNARY_OPERATORS: dict[str, Callable[[float], float]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": safe_tan,
    "exp": safe_exp,
    "log": safe_log,
    "plog": plog,  # Protected log
    "sqrt": safe_sqrt,
    "psqrt": psqrt,  # Protected sqrt
    "abs": np.abs,
    "neg": lambda x: -x,
    "inv": safe_inv,
    "square": safe_square,
    "cube": safe_cube,
    "sinh": safe_sinh,
    "cosh": safe_cosh,
    "tanh": np.tanh,
    "bessel_j0": scipy_special.j0,  # Bessel function of first kind, order 0
    "bessel_j1": scipy_special.j1,  # Bessel function of first kind, order 1
    "gamma": scipy_special.gamma,   # Gamma function (extends factorials)
    "prime_pi": safe_prime_pi,      # Prime-counting function π(x)
}

BINARY_OPERATORS: dict[str, Callable[[float, float], float]] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": safe_mul,
    "div": safe_div,
    "pow": safe_pow,
    "max": np.maximum,
    "min": np.minimum,
    "bitwise_xor": np.vectorize(safe_bitwise_xor),
    "bitwise_and": np.vectorize(safe_bitwise_and),
    "bitwise_or": np.vectorize(safe_bitwise_or),
    "lshift": np.vectorize(safe_lshift),
    "rshift": np.vectorize(safe_rshift),
}

# SymPy equivalents for symbolic conversion
SYMPY_UNARY = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "plog": lambda x: sp.log(sp.Abs(x) + 1e-10),
    "sqrt": sp.sqrt,
    "psqrt": lambda x: sp.sqrt(sp.Abs(x)),
    "abs": sp.Abs,
    "neg": lambda x: -x,
    "inv": lambda x: 1/x,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "bessel_j0": lambda x: sp.besselj(0, x),
    "bessel_j1": lambda x: sp.besselj(1, x),
    "prime_pi": sp.primepi,
}

SYMPY_BINARY: dict[str, Callable] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "pow": lambda x, y: x**y,
    "max": sp.Max,
    "min": sp.Min,
    # Use symbolic functions for bitwise ops to preserve semantics
    "bitwise_xor": sp.Function("bitwise_xor"),
    "bitwise_and": sp.Function("bitwise_and"),
    "bitwise_or": sp.Function("bitwise_or"),
    "lshift": sp.Function("lshift"),
    "rshift": sp.Function("rshift"),
}


@dataclass(eq=False)
class ExpressionNode:
    """A node in an expression tree.

    Attributes:
        node_type: Type of this node (CONSTANT, VARIABLE, UNARY_OP, BINARY_OP)
        value: For CONSTANT: the numeric value; for VARIABLE: the variable name;
               for operators: the operator name (e.g., 'add', 'sin')
        children: List of child nodes (empty for terminals, 1 for unary, 2 for binary)
        parent: Reference to parent node (None for root)
    """

    node_type: NodeType
    value: Any
    children: list[ExpressionNode] = field(default_factory=list)
    parent: ExpressionNode | None = field(default=None, repr=False)

    def __post_init__(self):
        """Set parent references for children."""
        for child in self.children:
            child.parent = self

    @property
    def arity(self) -> int:
        """Number of children this node should have."""
        if self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE):
            return 0
        elif self.node_type == NodeType.UNARY_OP:
            return 1
        else:  # BINARY_OP
            return 2

    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal (leaf) node."""
        return self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE)

    def evaluate(self, variables: dict[str, float | np.ndarray]) -> float | np.ndarray:
        """Evaluate this subtree with given variable values.

        Args:
            variables: Dict mapping variable names to their values

        Returns:
            Computed value (scalar or array)
        """
        if self.node_type == NodeType.CONSTANT:
            # Return constant, broadcast if needed
            if isinstance(next(iter(variables.values()), 0), np.ndarray):
                return np.full_like(next(iter(variables.values())), self.value)
            return self.value

        elif self.node_type == NodeType.VARIABLE:
            return variables.get(self.value, 0.0)

        elif self.node_type == NodeType.UNARY_OP:
            child_val = self.children[0].evaluate(variables)
            op_func = UNARY_OPERATORS.get(self.value)
            if op_func is None:
                raise ValueError(f"Unknown unary operator: {self.value}")
            try:
                result = op_func(child_val)
                # Handle NaN/Inf
                # For complex numbers, isfinite checks both real/imag
                # We do NOT use nan_to_num with scalar defaults because it might cast complex to real
                if isinstance(result, np.ndarray):
                    # Replace NaNs with 0, Infs with large number
                    # This works for complex arrays in recent numpy
                    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
                elif not np.isfinite(result):
                    result = 0.0
                return result
            except Exception:
                return 0.0

        else:  # BINARY_OP
            left_val = self.children[0].evaluate(variables)
            right_val = self.children[1].evaluate(variables)
            op_func = BINARY_OPERATORS.get(self.value)
            if op_func is None:
                raise ValueError(f"Unknown binary operator: {self.value}")
            try:
                result = op_func(left_val, right_val)
                # Handle NaN/Inf
                if isinstance(result, np.ndarray):
                    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
                elif not np.isfinite(result):
                    result = 0.0
                return result
            except Exception:
                return 0.0

    def to_sympy(self, symbols: dict[str, sp.Symbol]) -> sp.Expr:
        """Convert this subtree to a SymPy expression.

        Args:
            symbols: Dict mapping variable names to SymPy symbols

        Returns:
            SymPy expression
        """
        if self.node_type == NodeType.CONSTANT:
            # Skip rationalization for very large or very small numbers
            if abs(self.value) > 1e6 or (abs(self.value) < 1e-6 and self.value != 0):
                return sp.Float(self.value)
            # Try to rationalize the constant for small values
            try:
                rational = sp.nsimplify(self.value, tolerance=1e-6, rational=True)
                if abs(float(rational) - self.value) < 1e-6:
                    return rational
            except Exception:
                pass
            return sp.Float(self.value)

        elif self.node_type == NodeType.VARIABLE:
            return symbols.get(self.value, sp.Symbol(self.value))

        elif self.node_type == NodeType.UNARY_OP:
            child_expr = self.children[0].to_sympy(symbols)
            op_func = SYMPY_UNARY.get(self.value)
            if op_func is None:
                raise ValueError(f"No SymPy equivalent for: {self.value}")
            return op_func(child_expr)

        else:  # BINARY_OP
            left_expr = self.children[0].to_sympy(symbols)
            right_expr = self.children[1].to_sympy(symbols)
            op_func = SYMPY_BINARY.get(self.value)
            if op_func is None:
                raise ValueError(f"No SymPy equivalent for: {self.value}")
            return op_func(left_expr, right_expr)

    def copy_subtree(self) -> ExpressionNode:
        """Create a deep copy of this subtree."""
        new_node = ExpressionNode(
            node_type=self.node_type,
            value=self.value,
            children=[child.copy_subtree() for child in self.children],
            parent=None,
        )
        return new_node

    def count_nodes(self) -> int:
        """Count total nodes in this subtree."""
        return 1 + sum(child.count_nodes() for child in self.children)
    
    def calculate_weighted_complexity(self, weights: dict[str, float] | None = None, default_weight: float = 1.0) -> float:
        """Calculate weighted complexity of this subtree."""
        cost = default_weight
        
        # Determine cost of this node
        if self.node_type in (NodeType.UNARY_OP, NodeType.BINARY_OP):
            if weights:
                cost = weights.get(str(self.value), default_weight)
        elif self.node_type == NodeType.CONSTANT:
            cost = 1.0
        elif self.node_type == NodeType.VARIABLE:
            cost = 1.0
            
        # Recursive sum
        child_cost = sum(child.calculate_weighted_complexity(weights, default_weight) for child in self.children)
        return cost + child_cost

    def depth(self) -> int:
        """Calculate depth of this subtree."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def __str__(self) -> str:
        """String representation of this subtree."""
        if self.node_type == NodeType.CONSTANT:
            return f"{self.value:.6g}"
        elif self.node_type == NodeType.VARIABLE:
            return str(self.value)
        elif self.node_type == NodeType.UNARY_OP:
            return f"{self.value}({self.children[0]})"
        else:  # BINARY_OP
            op_symbol = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "pow": "^",
                "max": "max",
                "min": "min",
            }.get(self.value, self.value)
            if self.value in ("add", "sub", "mul", "div", "pow"):
                return f"({self.children[0]} {op_symbol} {self.children[1]})"
            else:
                return f"{op_symbol}({self.children[0]}, {self.children[1]})"


@dataclass
class ExpressionTree:
    """A complete expression tree representing a mathematical function.

    This is the main class for genetic programming symbolic regression.
    Supports evaluation, mutation, crossover, and conversion to symbolic form.

    Attributes:
        root: Root node of the expression tree
        variables: List of variable names (e.g., ['x', 'y'])
        fitness: Cached fitness value (lower is better)
        age: Generation when this tree was created
    """

    root: ExpressionNode
    variables: list[str] = field(default_factory=lambda: ["x"])
    fitness: float = field(default=float("inf"))
    age: int = field(default=0)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the expression tree on input data.

        Args:
            X: Input data of shape (n_samples,) for single variable
               or (n_samples, n_variables) for multiple variables

        Returns:
            Array of computed values, shape (n_samples,)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Build variables dict
        var_dict = {}
        for i, var_name in enumerate(self.variables):
            if i < X.shape[1]:
                var_dict[var_name] = X[:, i]
            else:
                var_dict[var_name] = np.zeros(X.shape[0])

        with np.errstate(all="ignore"):
            result = self.root.evaluate(var_dict)

        # Ensure result is array of correct shape
        if isinstance(result, (int, float, complex, np.number)):
            result = np.full(X.shape[0], result)

        return result

    def to_sympy(self) -> sp.Expr:
        """Convert to a SymPy expression (no simplification)."""
        symbols = {var: sp.Symbol(var) for var in self.variables}
        return self.root.to_sympy(symbols)

    def to_string(self) -> str:
        """Get string representation of the expression."""
        return str(self.root)

    def to_pretty_string(self) -> str:
        """Get a cleaned-up string representation."""
        try:
            # Skip simplification for complex trees to avoid hangs
            if self.complexity() > 20:
                return self.to_string()
            symbols = {var: sp.Symbol(var) for var in self.variables}
            expr = self.root.to_sympy(symbols)
            # Fix SymPy capitalization for Python compatibility
            s = str(expr)
            s = s.replace("Max", "max").replace("Min", "min")
            return s
        except Exception:
            return self.to_string()

    def copy(self) -> ExpressionTree:
        """Create a deep copy of this tree."""
        return ExpressionTree(
            root=self.root.copy_subtree(),
            variables=self.variables.copy(),
            fitness=self.fitness,
            age=self.age,
        )

    def complexity(self, weights: dict[str, float] | None = None, default_weight: float = 1.0) -> float:
        """Return the complexity of this tree.
        
        Args:
            weights: Optional dictionary of operator weights.
                     If None, returns simple node count (unweighted).
            default_weight: Weight for unknown operators/nodes (default 1.0)
        """
        if weights is None:
             return float(self.root.count_nodes())
        return self.root.calculate_weighted_complexity(weights, default_weight)

    def depth(self) -> int:
        """Return the depth of this tree."""
        return self.root.depth()

    def get_all_nodes(self) -> list[ExpressionNode]:
        """Get a flat list of all nodes in the tree."""
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    def get_random_node(self) -> ExpressionNode:
        """Get a random node from the tree."""
        nodes = self.get_all_nodes()
        return random.choice(nodes)

    def get_random_subtree(self) -> ExpressionNode:
        """Get a random non-root subtree from the tree."""
        nodes = self.get_all_nodes()
        non_root = [n for n in nodes if n.parent is not None]
        if not non_root:
            return self.root
        return random.choice(non_root)

    def replace_subtree(self, old_node: ExpressionNode, new_subtree: ExpressionNode):
        """Replace a subtree with a new one.

        Args:
            old_node: Node to replace
            new_subtree: New subtree to insert
        """
        if old_node.parent is None:
            # Replacing root
            self.root = new_subtree
            new_subtree.parent = None
        else:
            parent = old_node.parent
            # Find by identity, not equality
            idx = None
            for i, child in enumerate(parent.children):
                if child is old_node:
                    idx = i
                    break
            if idx is not None:
                parent.children[idx] = new_subtree
                new_subtree.parent = parent

    def fold_constants(self):
        """Recursively fold constant subtrees into single constant nodes.
        
        Example: sqrt(x^2 - 100/6) -> sqrt(x^2 - 16.66)
        This exposes the computed float value to optimization logic.
        """
        self.root = self._fold_recursive(self.root)
        self.root.parent = None

    def _fold_recursive(self, node: ExpressionNode) -> ExpressionNode:
        # 1. Fold children first (bottom-up)
        for i, child in enumerate(node.children):
            node.children[i] = self._fold_recursive(child)
            node.children[i].parent = node

        # 2. If node is operator and all children are constants, eval and replace
        if not node.is_terminal and all(c.node_type == NodeType.CONSTANT for c in node.children):
            try:
                # Create dummy var dict (constants don't need vars)
                val = node.evaluate({})
                # Return new constant node
                return ExpressionNode(NodeType.CONSTANT, val)
            except Exception:
                # If eval fails (e.g. div by zero), keep original structure
                return node
        
        # 3. Special Case: Bitwise Operator Constant Snapping
        # If we have bitwise_xor(x, 5.43), the 5.43 is effectively 5.
        # We snap it to nearest integer to make the output cleaner (e.g. "bitwise_xor(x, 5)").
        if node.node_type == NodeType.BINARY_OP and node.value in {
            "bitwise_xor", "bitwise_and", "bitwise_or", "lshift", "rshift"
        }:
            for child in node.children:
                if child.node_type == NodeType.CONSTANT:
                    try:
                        # Snap to nearest integer
                        child.value = float(round(child.value))
                    except:
                        pass

        return node

    @staticmethod
    def random_tree(
        variables: list[str],
        max_depth: int = 4,
        operators: list[str] | None = None,
        method: str = "grow",
    ) -> ExpressionTree:
        """Generate a random expression tree.

        Args:
            variables: List of variable names
            max_depth: Maximum tree depth
            operators: List of operator names to use (default: common set)
            method: 'grow' (variable depth) or 'full' (max depth for all branches)

        Returns:
            New random ExpressionTree
        """
        if operators is None:
            operators = ["add", "sub", "mul", "div", "sin", "cos", "exp", "square"]

        unary_ops = [op for op in operators if op in UNARY_OPERATORS]
        binary_ops = [op for op in operators if op in BINARY_OPERATORS]

        def build_node(depth: int) -> ExpressionNode:
            # Terminal probability increases with depth
            if depth >= max_depth or (
                method == "grow" and depth > 1 and random.random() < 0.3
            ):
                # Terminal node
                if random.random() < 0.5 and variables:
                    # Variable
                    return ExpressionNode(
                        node_type=NodeType.VARIABLE, value=random.choice(variables)
                    )
                else:
                    # Constant
                    const = random.choice(
                        [
                            random.uniform(-10, 10),
                            random.randint(-5, 5),
                            math.pi,
                            math.e,
                            0.5,
                            2.0,
                        ]
                    )
                    return ExpressionNode(node_type=NodeType.CONSTANT, value=const)
            else:
                # Operator node
                if unary_ops and (not binary_ops or random.random() < 0.3):
                    # Unary operator
                    op = random.choice(unary_ops)
                    child = build_node(depth + 1)
                    node = ExpressionNode(
                        node_type=NodeType.UNARY_OP, value=op, children=[child]
                    )
                    child.parent = node
                    return node
                else:
                    # Binary operator
                    op = random.choice(binary_ops) if binary_ops else "add"
                    left = build_node(depth + 1)
                    right = build_node(depth + 1)
                    node = ExpressionNode(
                        node_type=NodeType.BINARY_OP, value=op, children=[left, right]
                    )
                    left.parent = node
                    right.parent = node
                    return node

        root = build_node(1)
        return ExpressionTree(root=root, variables=variables)

    @staticmethod
    def from_sympy(
        expr: sp.Expr,
        variables: list[str],
    ) -> ExpressionTree:
        """Create an ExpressionTree from a SymPy expression (for Seeding).

        Args:
            expr: SymPy expression
            variables: List of allowed variable names

        Returns:
            ExpressionTree representing the expression
        """

        def _convert_node(node) -> ExpressionNode:
            # Handle Imaginary Unit I -> treat as 0.0 to strip complex parts from seeds
            if node == sp.I:
                 return ExpressionNode(NodeType.CONSTANT, 0.0)

            # 1. Constant
            if node.is_Number:
                try:
                    val = float(node)
                    return ExpressionNode(NodeType.CONSTANT, val)
                except Exception:
                    # e.g. infinity?
                    return ExpressionNode(NodeType.CONSTANT, 0.0)

            # 2. Variable
            if node.is_Symbol:
                name = str(node)
                if name in variables:
                    return ExpressionNode(NodeType.VARIABLE, name)
                else:
                    # Treat unknown symbol as variable anyway? Or error?
                    # For seeding, better be lenient or assume parameter
                    return ExpressionNode(NodeType.VARIABLE, name)

            # 3. Operations

            # SymPy Internals Mapping
            if node.is_Add:
                # SymPy Add is n-ary. We must chain binary adds.
                # (a+b+c) -> add(a, add(b, c))
                operands = node.args
                # Recursive chaining
                current = _convert_node(operands[0])
                for i in range(1, len(operands)):
                    rhs = _convert_node(operands[i])
                    # Create parent wrapper
                    parent = ExpressionNode(NodeType.BINARY_OP, "add", [current, rhs])
                    current.parent = parent
                    rhs.parent = parent
                    current = parent
                return current

            elif node.is_Mul:
                operands = node.args
                current = _convert_node(operands[0])
                for i in range(1, len(operands)):
                    rhs = _convert_node(operands[i])
                    parent = ExpressionNode(NodeType.BINARY_OP, "mul", [current, rhs])
                    current.parent = parent
                    rhs.parent = parent
                    current = parent
                return current

            elif node.is_Pow:
                base = _convert_node(node.base)
                exp = _convert_node(node.exp)
                # Check if it's actually sqrt?
                # SymPy usually represents sqrt(x) as Pow(x, 1/2).
                # But ExpressionTree might have "sqrt".
                # Standard pow is fine.
                parent = ExpressionNode(NodeType.BINARY_OP, "pow", [base, exp])
                base.parent = parent
                exp.parent = parent
                return parent

            elif isinstance(node, sp.Function) or hasattr(node, "func"):
                fname = node.func.__name__.lower()
                if fname == "max":
                    operands = node.args
                    current = _convert_node(operands[0])
                    for i in range(1, len(operands)):
                        rhs = _convert_node(operands[i])
                        parent = ExpressionNode(
                            NodeType.BINARY_OP, "max", [current, rhs]
                        )
                        current.parent = parent
                        rhs.parent = parent
                        current = parent
                    return current
                elif fname == "min":
                    operands = node.args
                    current = _convert_node(operands[0])
                    for i in range(1, len(operands)):
                        rhs = _convert_node(operands[i])
                        parent = ExpressionNode(
                            NodeType.BINARY_OP, "min", [current, rhs]
                        )
                        current.parent = parent
                        rhs.parent = parent
                        current = parent
                    return current

                child = _convert_node(node.args[0])
                if fname == "abs":
                    fname = "abs"

                if len(node.args) == 2:
                    # Binary function (e.g. bitwise_xor, lshift, rshift, pow, atan2)
                    child2 = _convert_node(node.args[1])
                    parent = ExpressionNode(NodeType.BINARY_OP, fname, [child, child2])
                    child.parent = parent
                    child2.parent = parent
                    return parent

                parent = ExpressionNode(NodeType.UNARY_OP, fname, [child])
                child.parent = parent
                return parent

            # Fallback
            return ExpressionNode(NodeType.CONSTANT, 0.0)

        root = _convert_node(expr)
        root.parent = None
        tree = ExpressionTree(root, variables)
        return tree

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"ExpressionTree({self.to_string()}, fitness={self.fitness:.6g})"
