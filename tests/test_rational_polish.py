from kalkulator_pkg.function_manager import find_function_from_data
import pytest

def test_rational_polish_weighted_avg():
    # Input: 0.318182 (approx 7/22)
    # f(0, 7.1) approx 4.84091 implies small non-zero intercept if fitting linearly exact, 
    # but Rational Polish should snap intercept to 0.
    
    data_points = [
        ([5.0, 7.1], 6.43182),
        ([0.0, 7.1], 4.84091),
        ([10.0, 10.0], 10.0),
    ]
    
    success, func_str, _, _ = find_function_from_data(data_points, param_names=["x", "y"])
    
    assert success
    print(f"\nResult: {func_str}")
    
    # Check for rational forms
    assert "7/22" in func_str, f"Expected 7/22, got {func_str}"
    assert "15/22" in func_str, f"Expected 15/22, got {func_str}"
    
    # Check intercept
    # It might include '+ 0'. Just checking it doesn't include the tiny float 7.58e-06
    assert "e-06" not in func_str
    assert "e-05" not in func_str
    
if __name__ == "__main__":
    try:
        test_rational_polish_weighted_avg()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error: {e}")
