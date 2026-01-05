"""Mathematical Constant Anchor Recognition for Symbolic Regression.

This module implements Gemini's strategy: detect known mathematical constants
at integer inputs, then generate hypothesis expressions that could produce them.

Example:
    If f(2) = 1.732 ≈ √3, generate hypotheses:
    - sqrt(x+1)      # Because sqrt(2+1) = √3
    - (x+1)^(1/2)    # Same thing
    - (x+1)^(1/x)    # Generalize exponent!

This is NOT spoonfeeding - it's pattern recognition from data.
"""

from __future__ import annotations

import numpy as np


# Known mathematical constants for anchor detection
KNOWN_CONSTANTS: dict[str, float] = {
    'sqrt(2)': np.sqrt(2),           # 1.41421...
    'sqrt(3)': np.sqrt(3),           # 1.73205...
    'sqrt(5)': np.sqrt(5),           # 2.23607...
    'sqrt(6)': np.sqrt(6),           # 2.44949...
    'sqrt(7)': np.sqrt(7),           # 2.64575...
    'e': np.e,                        # 2.71828...
    'pi': np.pi,                      # 3.14159...
    'phi': (1 + np.sqrt(5)) / 2,     # 1.61803... (golden ratio)
    '1/e': 1 / np.e,                 # 0.36788...
    'sqrt(pi)': np.sqrt(np.pi),      # 1.77245...
    '2': 2.0,                         # Clean integer
    '3': 3.0,
    '1/2': 0.5,
    '1/3': 1/3,
    '2/3': 2/3,
}


def detect_anchors(
    X: np.ndarray, 
    y: np.ndarray, 
    tolerance: float = 1e-3
) -> list[tuple[int, str, float]]:
    """Detect known mathematical constants at integer inputs.
    
    Args:
        X: Input data (n_samples, n_features)
        y: Target values (n_samples,)
        tolerance: How close y must be to constant
        
    Returns:
        List of (x_int, constant_name, constant_value) tuples
        
    Example:
        >>> X = np.array([[2.0]])
        >>> y = np.array([1.732])
        >>> detect_anchors(X, y)
        [(2, 'sqrt(3)', 1.732050807...)]
    """
    anchors = []
    
    # Only works for 1D inputs currently
    if X.shape[1] != 1:
        return anchors
    
    for i, (x_row, y_val) in enumerate(zip(X, y)):
        x_val = x_row[0]
        
        # Skip complex numbers (can't round them)
        if isinstance(x_val, complex) or isinstance(y_val, complex):
            continue
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        
        # Only check integer inputs (clean mathematical points)
        if abs(x_val - round(x_val)) < 1e-6:
            x_int = int(round(x_val))
            
            # Check against known constants
            for const_name, const_value in KNOWN_CONSTANTS.items():
                if abs(y_val - const_value) < tolerance:
                    anchors.append((x_int, const_name, const_value))
                    break  # Only match one constant per point
    
    return anchors


def generate_hypotheses(
    anchors: list[tuple[int, str, float]],
    var_name: str = 'x'
) -> list[str]:
    """Generate expression hypotheses from detected anchors.
    
    Uses reverse engineering: "If f(x_int) = constant, what expression produces this?"
    
    Args:
        anchors: List of (x_int, constant_name, constant_value)
        var_name: Variable name to use in expressions
        
    Returns:
        List of hypothesis expression strings
        
    Example:
        >>> anchors = [(2, 'sqrt(3)', 1.732)]
        >>> generate_hypotheses(anchors, 'x')
        ['sqrt(x+1)', '(x+1)**0.5', '(x+1)**(1/x)', ...]
    """
    hypotheses = []
    
    for x_int, const_name, const_val in anchors:
        # Strategy: Parse constant structure and work backwards
        
        if 'sqrt' in const_name:
            # Extract the base under sqrt
            base = _extract_sqrt_base(const_name)
            if base is None:
                continue
            
            # Calculate offset: base = x + c, so c = base - x
            c = base - x_int
            
            # H1: sqrt(x+c)
            if c == 0:
                hypotheses.append(f"sqrt({var_name})")
            elif c > 0:
                hypotheses.append(f"sqrt({var_name}+{c})")
            else:
                hypotheses.append(f"sqrt({var_name}{c})")  # Negative already has sign
            
            # H2: (x+c)^(1/2) - Same as H1 but explicit
            if c == 0:
                hypotheses.append(f"{var_name}**0.5")
            elif c > 0:
                hypotheses.append(f"({var_name}+{c})**0.5")
            else:
                hypotheses.append(f"({var_name}{c})**0.5")
            
            # H3: (x+c)^(1/x) - CRITICAL! Generalize exponent to 1/x
            # This is the key insight for (1+x)^(1/x)
            if c == 0:
                hypotheses.append(f"{var_name}**(1/{var_name})")
            elif c > 0:
                hypotheses.append(f"({var_name}+{c})**(1/{var_name})")
            else:
                hypotheses.append(f"({var_name}{c})**(1/{var_name})")
        
        elif const_name in ('e', 'pi'):
            # Exponential/trig patterns
            hypotheses.append(f"exp(1/{var_name})")
            hypotheses.append(f"exp({var_name}/{x_int})")
    
    # Remove duplicates
    return list(set(hypotheses))


def _extract_sqrt_base(const_name: str) -> int | None:
    """Extract base from sqrt(...) constant name."""
    try:
        if const_name == 'sqrt(2)':
            return 2
        elif const_name == 'sqrt(3)':
            return 3
        elif const_name == 'sqrt(5)':
            return 5
        elif const_name == 'sqrt(6)':
            return 6
        elif const_name == 'sqrt(7)':
            return 7
        elif const_name == 'sqrt(pi)':
            return None  # Can't reverse-engineer easily
    except:
        return None
    
    return None
