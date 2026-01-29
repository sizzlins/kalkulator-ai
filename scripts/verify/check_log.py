
import numpy as np
from numpy.lib import scimath

x = -4
y = -3
target = (-4)**(-3)
print(f"Target: {target}")

log_target = scimath.log(target)
print(f"log(target): {log_target}")

log_x = scimath.log(x)
y_log_x = y * log_x
print(f"y * log(x): {y_log_x}")

diff = log_target - y_log_x
print(f"Diff: {diff}")
print(f"Diff / (2pi*j): {diff / (2j * np.pi)}")

# Check user's specific point
# f(11, 12) = 313...
x2 = 11
y2 = 12
t2 = 11**12
print(f"\nPositive case:")
print(f"log(11^12): {scimath.log(t2)}")
print(f"12*log(11): {12*scimath.log(11)}")
