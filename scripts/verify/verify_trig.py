import numpy as np
from scipy.optimize import fsolve

def f1(x):
    return np.tan(x) - np.sin(np.cos(x))

def f2(x):
    return np.cos(x) - np.tan(x)

# User roots for tan(x)=sin(cos(x))
user_roots_1 = [
    -10.0515775450426, -5.65638572290634, -3.76839223786304, 
    0.626799584273244, 2.51479306931655, 6.90998489145283, 
    8.79797837649614, 13.1931701986324
]

# User roots for cos(x)=tan(x)
user_roots_2 = [
    -11.9001311818670, -10.0910173932620, -5.61694587468700, 
    -3.80783208608200, 0.666239432493000, 2.47535322109700, 
    6.94942473967200, 8.75853852827700
]

print("Checking tan(x) = sin(cos(x))")
for r in user_roots_1:
    res = f1(r)
    print(f"Room {r}: Residual {res: .2e} -> {'OK' if abs(res) < 1e-6 else 'FAIL'}")

print("\nChecking cos(x) = tan(x)")
for r in user_roots_2:
    res = f2(r)
    print(f"Room {r}: Residual {res: .2e} -> {'OK' if abs(res) < 1e-6 else 'FAIL'}")
