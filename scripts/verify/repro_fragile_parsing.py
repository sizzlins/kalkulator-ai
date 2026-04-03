
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.heuristics import _add_transcendental_interactions

def test_fragile_parsing():
    print("Testing Fragile Variable Parsing...")
    
    # Setup overlapping variable names
    variable_names = ['t', 't2']
    n_vars = 2
    
    # Create fake features representing unary transcendentals
    # We want to see if exp(t) and exp(t2) are correctly identified as single-variable features
    # and then interacted.
    
    # t features
    f_t = np.random.rand(10)
    name_t = "exp(t)"
    
    # t2 features
    f_t2 = np.random.rand(10)
    name_t2 = "exp(t2)" # Contains 't' as substring!
    
    features = [f_t, f_t2]
    feature_names = [name_t, name_t2]
    
    # Run interaction generation
    # If fragile parsing (substring) is used:
    # "exp(t2)" contains "t" -> vars_in_name += 1
    # "exp(t2)" contains "t2" -> vars_in_name += 1
    # Count = 2. It thinks it's already mixed "exp(t, t2)". Rejected.
    
    # If robust parsing (regex) is used:
    # "exp(t2)" matches \bt\b? No.
    # "exp(t2)" matches \bt2\b? Yes.
    # Count = 1. Accepted.
    
    # We expect an interaction "exp(t)*exp(t2)" to be generated.
    
    _add_transcendental_interactions(
        features, feature_names, n_vars, variable_names
    )
    
    print(f"Features after interaction: {len(feature_names)}")
    print(f"Feature Names: {feature_names}")
    
    expected = "exp(t)*exp(t2)"
    if expected in feature_names or "exp(t2)*exp(t)" in feature_names:
        print("SUCCESS: Found interaction 'exp(t)*exp(t2)'")
    else:
        print("FAILURE: Interaction 'exp(t)*exp(t2)' NOT found.")
        print("Bug confirmed: Fragile Parsing rejects overlapping variables.")

if __name__ == "__main__":
    test_fragile_parsing()
