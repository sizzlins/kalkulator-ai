"""Test linear recurrence detection for Lucas, Tribonacci, etc."""
import numpy as np

# Lucas sequence: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2)
lucas_seq = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]

# Tribonacci: T(0)=0, T(1)=0, T(2)=1, T(n)=T(n-1)+T(n-2)+T(n-3)
tribonacci_seq = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81]

# Pell: P(0)=0, P(1)=1, P(n)=2*P(n-1)+P(n-2)
pell_seq = [0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378]

from kalkulator_pkg.symbolic_regression.forensic_analysis import _detect_linear_recurrence

print("Testing Lucas sequence...")
X_lucas = np.array(range(len(lucas_seq))).reshape(-1, 1)
y_lucas = np.array(lucas_seq)
result_lucas = _detect_linear_recurrence(X_lucas, y_lucas, variable_names=["x"], verbose=True)
print(f"Result: {result_lucas}\n")

print("Testing Tribonacci sequence...")
X_trib = np.array(range(len(tribonacci_seq))).reshape(-1, 1)
y_trib = np.array(tribonacci_seq)
result_trib = _detect_linear_recurrence(X_trib, y_trib, variable_names=["x"], verbose=True)
print(f"Result: {result_trib}\n")

print("Testing Pell sequence...")
X_pell = np.array(range(len(pell_seq))).reshape(-1, 1)
y_pell = np.array(pell_seq)
result_pell = _detect_linear_recurrence(X_pell, y_pell, variable_names=["x"], verbose=True)
print(f"Result: {result_pell}\n")
