import sympy as sp
val = sp.sympify("-(-15)**(15/16)/15")
num_val = sp.N(val, 20)
print(f"NumVal Type: {type(num_val)}")
print(f"Is Number: {getattr(num_val, 'is_Number', 'Missing')}")
try:
    c = complex(num_val)
    print(f"Complex conversion: {c}")
except Exception as e:
    print(f"Complex conversion failed: {e}")
