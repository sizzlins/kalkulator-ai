import sympy as sp
from sympy import I

# Case from user
# 2.71828182845905 is roughly e
e_val = sp.Float('2.71828182845905')
pi_val = sp.Float('3.14159265358979')
# expr: e^(i*pi) + 1
expr = e_val**(I * pi_val) + 1

print(f"Expr: {expr}")
num_val = sp.N(expr, 15)
print(f"Num val: {num_val}")
print(f"Type: {type(num_val)}")
print(f"is_Number: {getattr(num_val, 'is_Number', False)}")
print(f"Abs: {abs(num_val)}")

if abs(num_val) < 1e-9:
    print("Should return '0'")
else:
    print("Should NOT return '0'")

imag_part = abs(sp.im(num_val))
print(f"Imag part: {imag_part}")
