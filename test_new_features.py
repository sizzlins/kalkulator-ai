"""Test new product features: x*sin(x), cosh(x), etc."""
import sys

sys.path.insert(0, ".")

if __name__ == "__main__":
    import math

    from kalkulator_pkg.function_manager import find_function_from_data

    # Test 1: x*sin(x) - Growing wave
    print("Test 1: g(x) = x*sin(x)")
    print("  Data: g(0)=0, g(pi/2)=pi/2, g(pi)=0, g(1.5*pi)=-1.5*pi, g(2*pi)=0")
    
    data_points = [
        (["0"], "0"),
        ([str(math.pi/2)], str(math.pi/2)),  # pi/2 * sin(pi/2) = pi/2 * 1 = pi/2
        ([str(math.pi)], "0"),                # pi * sin(pi) = 0
        ([str(1.5*math.pi)], str(-1.5*math.pi)),  # 1.5pi * sin(1.5pi) = 1.5pi * (-1) = -1.5pi
        ([str(2*math.pi)], "0"),             # 2pi * sin(2pi) = 0
    ]
    
    try:
        success, func_str, factored, error = find_function_from_data(data_points, ["x"])
        if success:
            print(f"  RESULT: g(x) = {func_str}")
            if "sin" in func_str and "*sin" in func_str:
                print("  ✓ Contains x*sin(x) pattern!")
            else:
                print("  Note: May need more/better data points")
        else:
            print(f"  FAILURE: {error}")
    except Exception as e:
        print(f"  CRASH: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: cosh(x) - Catenary
    print("\nTest 2: y(x) = cosh(x)")
    print("  Data: y(0)=1, y(1)=1.543, y(-1)=1.543, y(2)=3.762, y(-2)=3.762")
    
    data_points_2 = [
        (["0"], str(math.cosh(0))),       # cosh(0) = 1
        (["1"], str(math.cosh(1))),       # cosh(1) ≈ 1.543
        (["-1"], str(math.cosh(-1))),     # cosh(-1) ≈ 1.543
        (["2"], str(math.cosh(2))),       # cosh(2) ≈ 3.762
        (["-2"], str(math.cosh(-2))),     # cosh(-2) ≈ 3.762
    ]
    
    try:
        success, func_str, factored, error = find_function_from_data(data_points_2, ["x"])
        if success:
            print(f"  RESULT: y(x) = {func_str}")
            if "cosh" in func_str:
                print("  ✓ Found cosh(x)!")
        else:
            print(f"  FAILURE: {error}")
    except Exception as e:
        print(f"  CRASH: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check feature list
    print("\nTest 3: Verifying features are generated")
    import numpy as np

    from kalkulator_pkg.function_finder_advanced import generate_candidate_features
    
    X = np.array([[0.0], [1.0], [2.0]])
    _, names = generate_candidate_features(X, ["x"])
    
    new_features = ["x*sin(x)", "x*cos(x)", "sinh(x)", "cosh(x)", "x*log(x)"]
    for feat in new_features:
        if feat in names:
            print(f"  ✓ {feat} is in feature set")
        else:
            print(f"  ✗ {feat} is MISSING from feature set")
