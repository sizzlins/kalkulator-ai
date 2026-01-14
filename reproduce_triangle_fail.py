
import numpy as np
from kalkulator_pkg.symbolic_regression.forensic_analysis import generate_pattern_seeds

# Data approximated from the user's screenshot/log
# f(4.5)=0.5, f(4.4)=0.4, ..., f(4.0)=0, f(3.9)=0.1, ...
# This looks like abs(x - round(x)) or similar
# Let's generate perfect triangle wave data 
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = np.abs(X - np.round(X))

print("Testing generate_pattern_seeds with Triangle Wave data...")
seeds = generate_pattern_seeds(X, y, ["x"], verbose=True)

print(f"\nSeeds found: {seeds}")

if any("abs" in s.lower() and "round" in s.lower() for s in seeds):
    print("SUCCESS: Triangle wave detected.")
else:
    print("FAILURE: Triangle wave NOT detected.")
