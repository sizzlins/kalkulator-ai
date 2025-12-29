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
    # Use scimath.power to handle negative bases -> complex results
    # Still clip exponent to avoid overflow, but allow complex base
    # Output clipping is tricky for complex, we clip magnitude?
    res = np.lib.scimath.power(x, np.clip(y, -10, 10))
    # Complex clipping (magnitude cap)
    # If magnitude > 1e100, scale it down
    mag = np.abs(res)
    mask = mag > 1e100
    if np.any(mask):
        # Create scale factor: 1.0 where safe, 1e100/mag where unsafe
        scale = np.ones_like(mag)
        scale[mask] = 1e100 / mag[mask]
        res = res * scale
    return res


def safe_sinh(x):
    return np.sinh(np.clip(x, -700, 700))


def safe_cosh(x):
    return np.cosh(np.clip(x, -700, 700))


def safe_square(x):
    return np.clip(x * x, -1e100, 1e100)


def safe_cube(x):
    return np.clip(x * x * x, -1e100, 1e100)


UNARY_OPERATORS: dict[str, Callable[[float], float]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": safe_tan,
    "exp": safe_exp,
    "log": safe_log,
    "sqrt": safe_sqrt,
    "abs": np.abs,
    "neg": lambda x: -x,
    "inv": safe_inv,
    "square": safe_square,
    "cube": safe_cube,
    "sinh": safe_sinh,
    "cosh": safe_cosh,
    "tanh": np.tanh,
}

BINARY_OPERATORS: dict[str, Callable[[float, float], float]] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": safe_mul,
    "div": safe_div,
    "pow": safe_pow,
    "max": np.maximum,
    "min": np.minimum,
}

# SymPy equivalents for symbolic conversion
SYMPY_UNARY: dict[str, Callable] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "neg": lambda x: -x,
    "inv": lambda x: 1 / x,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
}

SYMPY_BINARY: dict[str, Callable] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "pow": lambda x, y: x**y,
    "max": sp.Max,
    "min": sp.Min,
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

    def complexity(self) -> int:
        """Return the complexity (number of nodes) of this tree."""
        return self.root.count_nodes()

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
