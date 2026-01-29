
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.symbolic_regression import forensic_analysis

def test_sin_detection():
    print("Testing sin(x) detection...")
    # Generate data: f(x,y) = y * sin(x)
    # Range [-10, 10]
    X = []
    y = []
    
    # Generate random points + some zeros at k*pi
    for _ in range(100):
        x = np.random.uniform(-10, 10)
        y_val = np.random.uniform(-5, 5)
        X.append([x])
        y.append(y_val * np.sin(x))
        
    # Explicitly add zeros at k*pi to simulate what 'y*sin(x)' produces
    # Because floating point sin(pi) is approx 1e-16 (close to zero but not exact 0)
    # Our data generation loop above produces ~1e-16.
    # But let's add some "perfect" zeros at random y values
    for k in range(-3, 4):
        x = k * np.pi
        y_val = np.random.uniform(-5, 5)
        X.append([x])
        y.append(0.0) # Exactly 0
        
    X = np.array(X)
    y = np.array(y)
    
    seeds = forensic_analysis._detect_zero_patterns(X, y, variable_names=["x"], verbose=True)
    print("Found seeds:", seeds)
    
    assert "sin(x)" in seeds, "Failed to detect sin(x)"
    
def test_cos_detection():
    print("\nTesting cos(x) detection...")
    # Generate data: f(x,y) = y * cos(x)
    X = []
    y = []
    
    for k in range(-3, 4):
        x = k * np.pi + np.pi/2
        y_val = np.random.uniform(-5, 5)
        X.append([x])
        y.append(0.0)
        
    # Add random noise points
    for _ in range(50):
        x = np.random.uniform(-10, 10)
        X.append([x])
        y.append(np.random.uniform(-5, 5)) # Random y, cos will make it non-zero mostly
        
    X = np.array(X)
    y = np.array(y)
    
    seeds = forensic_analysis._detect_zero_patterns(X, y, variable_names=["x"], verbose=True)
    print("Found seeds:", seeds)
    
    assert "cos(x)" in seeds, "Failed to detect cos(x)"

if __name__ == "__main__":
    test_sin_detection()
    test_cos_detection()
    print("\nSuccess!")
