import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from kalkulator_pkg.symbolic_regression.expression_tree import (
    ExpressionTree, ExpressionNode, NodeType,
    safe_sqrt, safe_exp
)

def verify_tree():
    print("--- Verifying Expression Tree Complex Eval ---")
    
    # Test 1: Manual operator chain
    print("\n[Test 1] Manual operator chain:")
    x_val = np.array([-4.0])
    print(f"  x = {x_val}, dtype={x_val.dtype}")
    
    step1 = safe_sqrt(x_val)
    print(f"  safe_sqrt(x) = {step1}, dtype={step1.dtype}")
    
    step2 = safe_exp(step1)
    print(f"  safe_exp(sqrt(x)) = {step2}, dtype={step2.dtype}")
    
    # Test 2: Manual operator chain with complex input
    print("\n[Test 2] Manual operator chain with COMPLEX input:")
    x_complex = np.array([-4.0+0j], dtype=np.complex128)
    print(f"  x = {x_complex}, dtype={x_complex.dtype}")
    
    step1c = safe_sqrt(x_complex)
    print(f"  safe_sqrt(x) = {step1c}, dtype={step1c.dtype}")
    
    step2c = safe_exp(step1c)
    print(f"  safe_exp(sqrt(x)) = {step2c}, dtype={step2c.dtype}")
    
    # Test 3: ExpressionTree evaluate with FLOAT input
    print("\n[Test 3] ExpressionTree.evaluate with FLOAT input:")
    node_x = ExpressionNode(NodeType.VARIABLE, "x")
    node_sqrt = ExpressionNode(NodeType.UNARY_OP, "sqrt", children=[node_x])
    node_exp = ExpressionNode(NodeType.UNARY_OP, "exp", children=[node_sqrt])
    tree = ExpressionTree(node_exp)
    print(f"  Tree: {tree}")
    
    X_float = np.array([[-4.0]])  # shape (1, 1), float64
    result_float = tree.evaluate(X_float)
    print(f"  Result: {result_float}, dtype={result_float.dtype}")
    
    expected = np.exp(np.lib.scimath.sqrt(np.array([-4.0])))
    print(f"  Expected: {expected}")
    
    # Test 4: ExpressionTree evaluate with COMPLEX input
    print("\n[Test 4] ExpressionTree.evaluate with COMPLEX input:")
    tree2 = ExpressionTree(
        ExpressionNode(NodeType.UNARY_OP, "exp", children=[
            ExpressionNode(NodeType.UNARY_OP, "sqrt", children=[
                ExpressionNode(NodeType.VARIABLE, "x")
            ])
        ])
    )
    X_complex = np.array([[-4.0+0j]], dtype=np.complex128)  # shape (1, 1), complex128
    result_complex = tree2.evaluate(X_complex)
    print(f"  Result: {result_complex}, dtype={result_complex.dtype}")
    print(f"  Expected: {expected}")
    
    # Test 5: Manual RPN trace
    print("\n[Test 5] Manual RPN trace:")
    tree3 = ExpressionTree(
        ExpressionNode(NodeType.UNARY_OP, "exp", children=[
            ExpressionNode(NodeType.UNARY_OP, "sqrt", children=[
                ExpressionNode(NodeType.VARIABLE, "x")
            ])
        ])
    )
    # Force compile
    tree3._compile_rpn()
    print(f"  RPN stack: {[(op, str(p)[:50]) for op, p in tree3._rpn_stack]}")
    
    # Step through manually
    X_test = np.array([[-4.0]])
    stack = []
    for opcode, payload in tree3._rpn_stack:
        if opcode == 0:
            stack.append(payload)
            print(f"  CONST: push {payload}")
        elif opcode == 1:
            val = X_test[:, payload]
            stack.append(val)
            print(f"  VAR[{payload}]: push {val}, dtype={val.dtype}")
        elif opcode == 2:
            arg = stack.pop()
            result = payload(arg)
            stack.append(result)
            print(f"  UNARY({payload.__name__ if hasattr(payload, '__name__') else '?'}): {arg} -> {result}, dtype={result.dtype if hasattr(result, 'dtype') else type(result)}")
        elif opcode == 3:
            right = stack.pop()
            left = stack.pop()
            result = payload(left, right)
            stack.append(result)
            print(f"  BINARY: {left} op {right} -> {result}")
    
    print(f"  Final: {stack[0]}, dtype={stack[0].dtype if hasattr(stack[0], 'dtype') else type(stack[0])}")

if __name__ == "__main__":
    verify_tree()
