"""Debug script to trace why integer pattern detection fails for (x³+1)/(x³-1)"""

import numpy as np
import fractions

# Simulate the user's data - focusing on key points
data_points = [
    (2.0, 1.28571428571429),   # 9/7 = (2³+1)/(2³-1)
    (3.0, 1.07692307692308),   # 28/26 = (3³+1)/(3³-1)  
    (4.0, 1.03174603174603),   # 65/63 = (4³+1)/(4³-1)
    (5.0, 1.01612903225806),   # 126/124 = (5³+1)/(5³-1)
    (-1.0, 0.0),              # Zero point
]

X = np.array([[dp[0]] for dp in data_points])
y = np.array([dp[1] for dp in data_points])

print("="*60)
print("DEBUGGING INTEGER PATTERN DETECTION")
print("="*60)

x_flat = X.flatten()
print(f"\nX shape: {X.shape}")
print(f"X.ndim: {X.ndim}")
print(f"x_flat: {x_flat}")
print(f"y: {y}")

# Filter check
print(f"\n--- Filtering for integer inputs ---")
for i, x_val in enumerate(x_flat):
    is_complex = np.iscomplex(x_val)
    print(f"  x_val={x_val}, is_complex={is_complex}")
    
    if is_complex:
        print(f"    SKIPPED: complex")
        continue
        
    try:
        real_val = float(x_val.real if hasattr(x_val, 'real') else x_val)
        is_integer = abs(real_val - round(real_val)) < 1e-9
        in_range = abs(real_val) > 1 and abs(real_val) < 10
        print(f"    real_val={real_val}, is_integer={is_integer}, in_range={in_range}")
        
        if is_integer and in_range:
            print(f"    ✅ ACCEPTED as integer input")
            
            # Now check the y value
            y_val = y[i]
            print(f"    y_val={y_val}")
            
            # Fraction conversion
            frac = fractions.Fraction(y_val).limit_denominator(1000)
            diff = abs(float(frac) - y_val)
            print(f"    Fraction: {frac} (diff={diff:.10f})")
            
            if diff > 1e-6:
                print(f"    ❌ SKIPPED: fraction diff too large")
                continue
                
            num = frac.numerator
            den = frac.denominator
            print(f"    num={num}, den={den}")
            
            # Check powers
            x_int = int(round(real_val))
            for n in [1, 2, 3]:
                x_pow = x_int ** n
                
                # Check numerator
                num_rel = None
                if num == x_pow: num_rel = f"x^{n}"
                elif num == x_pow + 1: num_rel = f"(x^{n} + 1)"
                elif num == x_pow - 1: num_rel = f"(x^{n} - 1)"
                
                # Check denominator
                den_rel = None
                if den == x_pow: den_rel = f"x^{n}"
                elif den == x_pow + 1: den_rel = f"(x^{n} + 1)"
                elif den == x_pow - 1: den_rel = f"(x^{n} - 1)"
                
                print(f"    n={n}: x^n={x_pow}, num_rel={num_rel}, den_rel={den_rel}")
                
                if num_rel and den_rel:
                    print(f"    🎯 FOUND PATTERN: {num_rel} / {den_rel}")
        else:
            print(f"    ❌ SKIPPED: not integer or out of range")
            
    except Exception as e:
        print(f"    ❌ EXCEPTION: {e}")

print("\n" + "="*60)
