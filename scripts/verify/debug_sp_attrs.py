import sympy as sp
try:
    print(f"sp.Relational: {sp.Relational}")
except AttributeError as e:
    print(f"Error accessing sp.Relational: {e}")
    
try:
    print(f"sp.core.relational.Relational: {sp.core.relational.Relational}")
except AttributeError as e:
    print(f"Error accessing sp.core.relational.Relational: {e}")

try:
    print(f"sp.MatrixBase: {sp.MatrixBase}")
except AttributeError as e:
    print(f"Error accessing sp.MatrixBase: {e}")
