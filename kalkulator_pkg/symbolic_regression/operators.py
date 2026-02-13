"""Genetic operators for evolving expression trees.

This module implements mutation and crossover operators for genetic programming:
- Point mutation: Replace a node with another of same arity
- Subtree mutation: Replace a subtree with a random new one
- Hoist mutation: Replace tree with one of its subtrees (simplification)
- Constant mutation: Perturb constant values
- Crossover: Swap subtrees between two parent trees
"""

from __future__ import annotations

import math
import random

import numpy as np

from .expression_tree import BINARY_OPERATORS
from .expression_tree import UNARY_OPERATORS
from .expression_tree import ExpressionNode
from .expression_tree import ExpressionTree
from .expression_tree import NodeType


def point_mutation(
    tree: ExpressionTree, mutation_rate: float = 0.1, operators: list[str] | None = None
) -> ExpressionTree:
    """Point mutation: Replace operators/terminals with others of same arity.

    Args:
        tree: Tree to mutate
        mutation_rate: Probability of mutating each node
        operators: Allowed operators

    Returns:
        Mutated tree (new copy)
    """
    if operators is None:
        operators = ["add", "sub", "mul", "div", "sin", "cos", "exp", "square"]

    unary_ops = [op for op in operators if op in UNARY_OPERATORS]
    binary_ops = [op for op in operators if op in BINARY_OPERATORS]

    new_tree = tree.copy()

    for node in new_tree.get_all_nodes():
        if random.random() > mutation_rate:
            continue

        if node.node_type == NodeType.CONSTANT:
            # Mutate constant value
            if random.random() < 0.5:
                # Small perturbation
                node.value *= 1 + random.gauss(0, 0.1)
            else:
                # Replace with new random constant
                node.value = random.choice(
                    [
                        random.uniform(-10, 10),
                        random.randint(-5, 5),
                        math.pi,
                        math.e,
                        0.5,
                        2.0,
                    ]
                )

        elif node.node_type == NodeType.VARIABLE:
            # Replace with different variable
            if len(tree.variables) > 1:
                other_vars = [v for v in tree.variables if v != node.value]
                if other_vars:
                    node.value = random.choice(other_vars)

        elif node.node_type == NodeType.UNARY_OP:
            # Replace with different unary operator
            if unary_ops:
                node.value = random.choice(unary_ops)

        elif node.node_type == NodeType.BINARY_OP:
            # INVERSE-AWARE MUTATIONS for pow operator
            # This enables discovery of complex functions like (1+x)^(1/x)
            if node.value == "pow" and len(node.children) == 2 and random.random() < 0.3:
                # 30% chance to try inverse exponent mutation
                base_node, exp_node = node.children
                
                # Strategy 1: If exponent is a constant, try reciprocal
                if exp_node.node_type == NodeType.CONSTANT:
                    # x^a → x^(1/a), but guard against zero
                    if abs(exp_node.value) > 0.01:  # Defensive: avoid division by zero
                        exp_node.value = 1.0 / exp_node.value
                
                # Strategy 2: If exponent is a variable, wrap in reciprocal
                elif exp_node.node_type == NodeType.VARIABLE:
                    # x^y → x^(1/y): create division node
                    inv_node = ExpressionNode(
                        node_type=NodeType.BINARY_OP,
                        value="div",
                        children=[
                            ExpressionNode(NodeType.CONSTANT, 1.0),
                            exp_node.copy_subtree()
                        ]
                    )
                    node.children[1] = inv_node
            else:
                # Original behavior: replace with different binary operator
                if binary_ops:
                    node.value = random.choice(binary_ops)


    return new_tree


def subtree_mutation(
    tree: ExpressionTree, max_depth: int = 3, operators: list[str] | None = None
) -> ExpressionTree:
    """Subtree mutation: Replace a random subtree with a new random one.

    Args:
        tree: Tree to mutate
        max_depth: Maximum depth of new subtree
        operators: Allowed operators

    Returns:
        Mutated tree (new copy)
    """
    new_tree = tree.copy()

    # Pick a random node to replace
    target_node = new_tree.get_random_node()

    # Generate new random subtree
    new_subtree = ExpressionTree.random_tree(
        variables=tree.variables,
        max_depth=max_depth,
        operators=operators,
        method="grow",
    ).root

    # Replace
    new_tree.replace_subtree(target_node, new_subtree)

    return new_tree


def hoist_mutation(tree: ExpressionTree) -> ExpressionTree:
    """Hoist mutation: Replace tree with one of its subtrees (simplification).

    This operator promotes a subtree to become the new root, effectively
    simplifying the tree. Helps prevent bloat.

    Args:
        tree: Tree to mutate

    Returns:
        Mutated tree (new copy)
    """
    new_tree = tree.copy()

    # Get all non-root nodes
    nodes = new_tree.get_all_nodes()
    non_root = [n for n in nodes if n.parent is not None]

    if not non_root:
        return new_tree

    # Pick a random subtree to hoist
    hoisted = random.choice(non_root)

    # Make it the new root
    hoisted_copy = hoisted.copy_subtree()
    new_tree.root = hoisted_copy

    return new_tree


def shrink_mutation(tree: ExpressionTree) -> ExpressionTree:
    """Shrink mutation: Replace a random subtree with a terminal.

    Similar to hoist but replaces with a simple terminal (variable or constant).

    Args:
        tree: Tree to mutate

    Returns:
        Mutated tree (new copy)
    """
    new_tree = tree.copy()

    # Pick a random non-terminal node
    nodes = new_tree.get_all_nodes()
    non_terminals = [n for n in nodes if not n.is_terminal]

    if not non_terminals:
        return new_tree

    target = random.choice(non_terminals)

    # Create terminal replacement
    if random.random() < 0.5 and tree.variables:
        replacement = ExpressionNode(
            node_type=NodeType.VARIABLE, value=random.choice(tree.variables)
        )
    else:
        replacement = ExpressionNode(
            node_type=NodeType.CONSTANT, value=random.uniform(-5, 5)
        )

    new_tree.replace_subtree(target, replacement)

    return new_tree



def insert_mutation(
    tree: ExpressionTree, operators: list[str] | None = None
) -> ExpressionTree:
    """Insert mutation: Insert a random unary operator above a random node.
    
    This allows evolving nested structures like floor(expr) from expr.
    
    Args:
        tree: Tree to mutate
        operators: Allowed operators
        
    Returns:
        Mutated tree (new copy)
    """
    if operators is None:
        operators = ["sin", "cos", "exp", "square", "floor", "trunc"]
        
    unary_ops = [op for op in operators if op in UNARY_OPERATORS]
    if not unary_ops:
        return tree.copy()
        
    new_tree = tree.copy()
    
    # Pick a random node to wrap
    target = new_tree.get_random_node()
    
    # Create wrapper node
    new_op = random.choice(unary_ops)
    wrapper = ExpressionNode(
        node_type=NodeType.UNARY_OP, 
        value=new_op, 
        children=[target.copy_subtree()]
    )
    
    # Replace in tree
    new_tree.replace_subtree(target, wrapper)
    
    return new_tree


def inject_variable_mutation(tree: ExpressionTree) -> ExpressionTree:
    """Inject a variable into a constant-only tree.
    
    This "Genetic Rescue" mutation prevents the population from converging
    to pure constants in transformed spaces (log, inverse) where the target
    variance is low and constants approximate well initially.
    
    Args:
        tree: Tree to mutate
        
    Returns:
        Mutated tree with at least one variable, or copy if already has variables.
    """
    # If tree already contains variables, no rescue needed
    if tree.contains_variables():
        return tree.copy()
    
    new_tree = tree.copy()
    
    # Find all constant nodes
    nodes = new_tree.get_all_nodes()
    constants = [n for n in nodes if n.node_type == NodeType.CONSTANT]
    
    if constants and tree.variables:
        # Pick a random constant to replace with a variable
        target = random.choice(constants)
        variable_name = random.choice(tree.variables)
        
        replacement = ExpressionNode(
            node_type=NodeType.VARIABLE,
            value=variable_name
        )
        
        new_tree.replace_subtree(target, replacement)
    
    return new_tree


def constant_optimization(

    tree: ExpressionTree,
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    iterations: int = 10,
) -> ExpressionTree:
    """Optimize constants in the tree using gradient-free optimization.

    Uses a simple hill-climbing approach to tune constant values.

    Args:
        tree: Tree to optimize
        X: Input data
        y: Target values
        learning_rate: Step size for perturbations
        iterations: Number of optimization iterations

    Returns:
        Tree with optimized constants
    """
    new_tree = tree.copy()
    
    # Pre-optimization: Fold constant subtrees (Agent Handoff Rule 5)
    # This prevents "rational jamming" where 100/6 is seen as two integers
    # instead of one float (16.66) that can be improved.
    new_tree.fold_constants()

    # Find all constant nodes
    nodes = new_tree.get_all_nodes()
    constants = [n for n in nodes if n.node_type == NodeType.CONSTANT and not getattr(n, 'locked', False)]

    if not constants:
        return new_tree

    # Current fitness
    try:
        pred = new_tree.evaluate(X)
        np.clip(pred, -1e100, 1e100, out=pred)  # Guard overflow
        diff = pred - y
        np.clip(diff, -1e100, 1e100, out=diff)  # Guard square
        current_mse = np.mean(np.abs(diff)**2)
    except (ValueError, OverflowError, TypeError, FloatingPointError):
        return new_tree

    for _ in range(iterations):
        for const_node in constants:
            original_value = const_node.value

            # Try perturbations
            for delta in [
                learning_rate,
                -learning_rate,
                learning_rate * 2,
                -learning_rate * 2,
            ]:
                const_node.value = original_value + delta * abs(original_value + 1e-10)

                try:
                    pred = new_tree.evaluate(X)
                    # Use Magnitude Squared Error for complex support
                    diff = pred - y
                    # np.abs handles complex magnitude correctly
                    squared_error = np.abs(diff) ** 2
                    # Clip to avoid overflow
                    np.clip(squared_error, 0, 1e100, out=squared_error)
                    new_mse = np.mean(squared_error)

                    if new_mse < current_mse:
                        current_mse = new_mse
                        original_value = const_node.value
                    else:
                        const_node.value = original_value
                except (OverflowError, ValueError, RuntimeWarning):
                    const_node.value = original_value
                except (OverflowError, ValueError, RuntimeWarning, FloatingPointError):
                    const_node.value = original_value

        # v4.0 Audit Remediation: Removed 5% "Integer Snapping" bias.
        # This heuristics was flagged as scientifically invalid (e.g. 3.04 != 3).
        # We now accept the optimizer's result as-is.

    return new_tree


def optimize_constants_bfgs(
    tree: ExpressionTree,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    sample_weight: np.ndarray | None = None
) -> ExpressionTree:
    """Optimize constants in the tree using BFGS (scipy.optimize).
    
    Much faster and more accurate than hill-climbing for finding optimal constants.
    Uses gradient-based optimization to find optimal values in few iterations.
    
    Args:
        tree: Tree to optimize
        X: Input data
        y: Target values
        max_iter: Maximum optimization iterations
        sample_weight: Optional weights for each sample
        
    Returns:
        Tree with optimized constants
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to hill-climbing if scipy not available
        return constant_optimization(tree, X, y)
    
    new_tree = tree.copy()
    new_tree.fold_constants()
    
    # NEW: Check for discrete primitives to prevent Livelock/Hang
    # Gradient-based optimizers (BFGS) get stuck on flat surfaces (floor, ceil, sign).
    all_nodes_check = new_tree.get_all_nodes()
    discrete_primitives = {'floor', 'ceil', 'ceiling', 'sign', 'signum', 'step', 'heaviside', 
                          'bitwise_or', 'bitwise_and', 'bitwise_xor', 'lshift', 'rshift', 
                          'factorial', 'gamma', 'lucas', 'fibonacci'}
    
    for node in all_nodes_check:
        if node.node_type in (NodeType.UNARY_OP, NodeType.BINARY_OP) and node.value in discrete_primitives:
             # print(f"[Debug] Skipping gradient optimization due to discrete primitive: {node.value}")
             return new_tree

    
    # Find all constant nodes
    nodes = new_tree.get_all_nodes()
    constants = [n for n in nodes if n.node_type == NodeType.CONSTANT and not getattr(n, 'locked', False)]
    
    if not constants:
        return new_tree
    
    # Extract initial values (real parts only for optimization)
    x0 = []
    for c in constants:
        val = c.value
        if isinstance(val, complex):
            x0.append(val.real)
        else:
            x0.append(float(val))
    x0 = np.array(x0)
    
    if len(x0) == 0:
        return new_tree
    
    # Define objective function (Weighted MSE)
    def objective(params):
        # Inject parameters into tree
        for i, c in enumerate(constants):
            # Preserve original type structure/imaginary part if needed
            # For now, we only optimize real components
            if isinstance(constants[i].value, complex):
                object.__setattr__(c, 'value', params[i] + 1j * constants[i].value.imag)
            else:
                object.__setattr__(c, 'value', params[i])
        
        # CRITICAL: Invalidate caches so evaluate() sees new constant values!
        new_tree._rpn_stack = None
        new_tree._numba_opcodes = None
        new_tree._numba_values = None
        new_tree._compiled_func = None
        
        try:
            pred = new_tree.evaluate(X)
            if not np.all(np.isfinite(pred)):
                return 1e10
                
            residuals = pred - y
            squared_error = np.abs(residuals) ** 2
            
            if sample_weight is not None:
                mse = np.average(squared_error, weights=sample_weight)
            else:
                mse = np.mean(squared_error)
                
            # Ensure purely real result (np.abs ensures positive real, but type might linger)
            return float(np.real(mse)) if np.isfinite(mse) else 1e10
        except (ValueError, OverflowError, TypeError, FloatingPointError):
            return 1e10
    
    # Run BFGS optimization
    try:
        # Debug: Check initial state
        # print("DEBUG: Starting BFGS")
        # print(f"DEBUG: x0 = {x0}")
        # print(f"DEBUG: Initial MSE = {objective(x0)}")
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',  # Bounded version, more robust
            options={'maxiter': max_iter, 'disp': False}
        )
        
        # print(f"DEBUG: Result success={result.success}, fun={result.fun}, x={result.x}")
        
        # Apply optimal values
        if result.success or result.fun < objective(x0):
            for i, c in enumerate(constants):
                if isinstance(constants[i].value, complex):
                     object.__setattr__(c, 'value', result.x[i] + 1j * constants[i].value.imag)
                else:
                    object.__setattr__(c, 'value', result.x[i])
            
            # CRITICAL FIX: Update fitness to reflect optimized constants
            # The tree was copied from parent, so it holds OLD fitness.
            # We must update it to the new optimized MSE.
            new_tree.fitness = float(np.real(result.fun)) if np.isfinite(result.fun) else None
            
            # CRITICAL FIX #2: Apply constant penalty if tree has no variables
            # This matches the penalty in calculate_fitness() - we can't skip it!
            if new_tree.fitness is not None and not new_tree.contains_variables():
                new_tree.fitness += 100.0
        else:
            # print("DEBUG: Optimization failed/worsened, reverting.")
            pass
                
        # v4.0 Audit Remediation: Removed 5% "Integer Snapping" bias.
        # This heuristics was flagged as scientifically invalid (e.g. 3.04 != 3).
        # We now accept the optimizer's result as-is.
        

    except (ValueError, RuntimeError) as e:
        # print(f"DEBUG: Optimization crashed: {e}")
        pass
    
    # Final cleanup of cache just in case
    new_tree._rpn_stack = None
    new_tree._numba_opcodes = None
    new_tree._numba_values = None
    
    return new_tree


def crossover(
    parent1: ExpressionTree, parent2: ExpressionTree, max_depth: int = 10
) -> tuple[ExpressionTree, ExpressionTree]:
    """Crossover: Swap subtrees between two parent trees.

    Creates two offspring by exchanging randomly selected subtrees.

    Args:
        parent1: First parent tree
        parent2: Second parent tree
        max_depth: Maximum allowed depth for offspring

    Returns:
        Tuple of two offspring trees
    """
    # Create copies
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Get crossover points
    point1 = offspring1.get_random_node()
    point2 = offspring2.get_random_node()

    # Copy subtrees
    subtree1 = point1.copy_subtree()
    subtree2 = point2.copy_subtree()

    # Swap
    offspring1.replace_subtree(point1, subtree2)
    offspring2.replace_subtree(point2, subtree1)

    # Check depth constraints and retry if needed
    if offspring1.depth() > max_depth:
        offspring1 = parent1.copy()
    if offspring2.depth() > max_depth:
        offspring2 = parent2.copy()

    return offspring1, offspring2


def tournament_selection(
    population: list[ExpressionTree], tournament_size: int = 5
) -> ExpressionTree:
    """Select an individual using tournament selection.

    Args:
        population: List of trees to select from
        tournament_size: Number of individuals in tournament

    Returns:
        Selected tree (not a copy)
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda t: t.fitness)


def lexicographic_selection(
    population: list[ExpressionTree], tournament_size: int = 5, epsilon: float = 1e-6
) -> ExpressionTree:
    """Lexicographic selection: prefer fitness, then simplicity.

    Among trees with similar fitness, prefer simpler ones.

    Args:
        population: List of trees to select from
        tournament_size: Number of individuals in tournament
        epsilon: Tolerance for considering fitnesses equal

    Returns:
        Selected tree
    """
    tournament = random.sample(population, min(tournament_size, len(population)))

    # Sort by fitness first
    tournament.sort(key=lambda t: t.fitness)

    # Among top performers (within epsilon), prefer simplest
    top_fitness = tournament[0].fitness
    top_group = [t for t in tournament if t.fitness <= top_fitness + epsilon]

    return min(top_group, key=lambda t: t.complexity())


def apply_mutation(
    tree: ExpressionTree,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    mutation_type: str = "random",
    operators: list[str] | None = None,
) -> ExpressionTree:
    """Apply a mutation to a tree.

    Args:
        tree: Tree to mutate
        X: Input data (for constant optimization)
        y: Target values (for constant optimization)
        mutation_type: Type of mutation ('point', 'subtree', 'hoist', 'shrink',
                       'constant', or 'random' for random selection)
        operators: Allowed operators

    Returns:
        Mutated tree
    """
    if mutation_type == "random":
        mutation_type = random.choice(["point", "subtree", "hoist", "shrink"])

    if mutation_type == "point":
        return point_mutation(tree, operators=operators)
    elif mutation_type == "subtree":
        return subtree_mutation(tree, operators=operators)
    elif mutation_type == "hoist":
        return hoist_mutation(tree)
    elif mutation_type == "shrink":
        return shrink_mutation(tree)
    elif mutation_type == "constant" and X is not None and y is not None:
        return constant_optimization(tree, X, y)
    else:
        return point_mutation(tree, operators=operators)
