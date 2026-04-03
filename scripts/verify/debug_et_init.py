
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree, ExpressionNode, NodeType
    
    print("Imported ExpressionTree successfully.")
    
    root = ExpressionNode(NodeType.CONSTANT, 1)
    print(f"Created Root: {root}")
    
    print("Attempting: ExpressionTree(root=root)")
    try:
        et = ExpressionTree(root=root)
        print(f"Success: {et}")
    except TypeError as e:
        print(f"Failed (TypeError): {e}")
    except Exception as e:
        print(f"Failed ({type(e).__name__}): {e}")

    print("Attempting: ExpressionTree.from_sympy(1)")
    try:
        if hasattr(ExpressionTree, 'from_sympy'):
             # Create dummy object behaving like SymPy Number to avoid importing sympy if possible
             class DummySymPy:
                 is_Number = True
                 is_Integer = True
                 is_Symbol = False
                 is_Add = False
                 is_Mul = False
                 is_Pow = False
                 args = []
                 def __int__(self): return 1
             
             et2 = ExpressionTree.from_sympy(DummySymPy())
             print(f"Success from_sympy: {et2}")
        else:
             print("ExpressionTree has no from_sympy method!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed from_sympy: {e}")

except Exception as e:
    print(f"Import failed: {e}")
