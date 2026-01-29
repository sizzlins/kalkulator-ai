import math
from kalkulator_pkg.symbolic_regression.symbolic_reconstruction import reconstruct_constant

def test_reconstruction():
    e = math.e
    pi = math.pi
    
    cases = [
        (e - pi, "e - pi"),
        (pi - e, "-e + pi"), # Or pi - e
        (-e * pi, "-1*e*pi"),
        (2 * e + 3, "2*e + 3"),
        (pi + 1, "pi + 1"),
        (0.423310825130746, None), # Random float? Wait, -0.4233 is e-pi approx
        (-0.423310825130746, "e - pi"), # Wait, e=2.718, pi=3.141, e-pi = -0.4233
        (-8.53973422267356, "-1*e*pi") # -e*pi = -8.5397
    ]
    
    print("Testing Symbolic Reconstruction:")
    for val, expected in cases:
        rec = reconstruct_constant(val, tolerance=1e-3)
        print(f"Value: {val:.5f} -> Reconstructed: {rec}")
        
test_reconstruction()
