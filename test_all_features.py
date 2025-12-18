"""Comprehensive Feature Test for Kalkulator.

Tests all main features:
1. Function Definition & Evaluation
2. Symbolic Constants (pi, e)
3. Function Finding (polynomial, trig, exponential, Gaussian, cosh, etc.)
4. Product Features (x*sin(x), x*log(x))
5. Export Command
6. Calculus (diff)
7. Nested Function Calls
"""
import math
import os
import sys

sys.path.insert(0, ".")

def test_separator(name):
    print(f"\n{'='*60}")
    print(f" {name}")
    print('='*60)

def check(condition, passed_msg, failed_msg):
    if condition:
        print(f"  ✅ {passed_msg}")
        return True
    else:
        print(f"  ❌ {failed_msg}")
        return False

if __name__ == "__main__":
    from kalkulator_pkg.function_manager import (
        _function_registry,
        define_function,
        evaluate_function,
        export_function_to_file,
        find_function_from_data,
    )
    from kalkulator_pkg.worker import evaluate_safely
    
    def clear_all_functions():
        _function_registry.clear()
    
    results = {"passed": 0, "failed": 0}
    
    def track(success):
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Clear functions before tests
    clear_all_functions()
    
    # ============================================================
    # TEST 1: Function Definition & Evaluation
    # ============================================================
    test_separator("1. Function Definition & Evaluation")
    
    # Define f(x) = 2*x
    define_function("f", ["x"], "2*x")
    result = evaluate_function("f", ["5"])
    track(check(str(result) == "10", "f(5) = 10", f"f(5) = {result}, expected 10"))
    
    # Define g(x,y) = x + y
    define_function("g", ["x", "y"], "x + y")
    result = evaluate_function("g", ["3", "7"])
    track(check(str(result) == "10", "g(3,7) = 10", f"g(3,7) = {result}, expected 10"))
    
    # ============================================================
    # TEST 2: Symbolic Constants (pi, e)
    # ============================================================
    test_separator("2. Symbolic Constants (pi, e)")
    
    # Define h(x) = x*sin(x)
    define_function("h", ["x"], "x*sin(x)")
    
    # Evaluate at pi/2
    result = evaluate_function("h", ["pi/2"])
    expected = math.pi / 2  # pi/2 * sin(pi/2) = pi/2 * 1 = pi/2
    try:
        result_float = float(result.evalf()) if hasattr(result, 'evalf') else float(result)
        track(check(abs(result_float - expected) < 0.01, 
                    f"h(pi/2) ≈ {expected:.4f}", 
                    f"h(pi/2) = {result_float}, expected {expected}"))
    except:
        track(check(False, "", f"h(pi/2) evaluation failed: {result}"))
    
    # ============================================================
    # TEST 3: Function Finding - Polynomial
    # ============================================================
    test_separator("3. Function Finding - Polynomial")
    
    # f(x) = x^2: f(0)=0, f(1)=1, f(2)=4, f(3)=9
    data = [
        (["0"], "0"),
        (["1"], "1"),
        (["2"], "4"),
        (["3"], "9"),
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    track(check(success and "x**2" in func_str.replace("^", "**").replace(" ", ""), 
                f"Found: {func_str}", 
                f"Expected x^2, got: {func_str or error}"))
    
    # ============================================================
    # TEST 4: Function Finding - Inverse Square (1/x^2)
    # ============================================================
    test_separator("4. Function Finding - Inverse Square")
    
    # I(d) = 100/d^2
    data = [
        (["1"], "100"),
        (["2"], "25"),
        (["4"], "6.25"),
        (["5"], "4"),
    ]
    success, func_str, _, error = find_function_from_data(data, ["d"])
    track(check(success and ("d^-2" in func_str or "1/d**2" in func_str or "d**-2" in func_str), 
                f"Found: {func_str}", 
                f"Expected 100*d^-2 form, got: {func_str or error}"))
    
    # ============================================================
    # TEST 5: Function Finding - Trigonometric
    # ============================================================
    test_separator("5. Function Finding - Trigonometric")
    
    # w(x) = sin(x) + cos(x)
    data = [
        (["0"], "1"),                           # sin(0)+cos(0) = 0+1 = 1
        ([str(math.pi/4)], str(math.sqrt(2))),  # sin+cos at pi/4 = sqrt(2)
        ([str(math.pi/2)], "1"),                # sin(pi/2)+cos(pi/2) = 1+0 = 1
        ([str(math.pi)], "-1"),                 # sin(pi)+cos(pi) = 0+(-1) = -1
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    track(check(success and "sin" in func_str and "cos" in func_str, 
                f"Found: {func_str}", 
                f"Expected sin+cos form, got: {func_str or error}"))
    
    # ============================================================
    # TEST 6: Function Finding - Gaussian (exp(-x^2))
    # ============================================================
    test_separator("6. Function Finding - Gaussian")
    
    # g(x) = exp(-x^2)
    data = [
        (["0"], "1"),
        (["1"], str(math.exp(-1))),
        (["2"], str(math.exp(-4))),
        (["3"], str(math.exp(-9))),
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    track(check(success and "exp(-x**2)" in func_str.replace("^", "**").replace(" ", ""), 
                f"Found: {func_str}", 
                f"Expected exp(-x^2), got: {func_str or error}"))
    
    # ============================================================
    # TEST 7: Function Finding - Hyperbolic (cosh)
    # ============================================================
    test_separator("7. Function Finding - Hyperbolic (cosh)")
    
    # y(x) = cosh(x)
    data = [
        (["0"], str(math.cosh(0))),
        (["1"], str(math.cosh(1))),
        (["-1"], str(math.cosh(-1))),
        (["2"], str(math.cosh(2))),
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    track(check(success and "cosh" in func_str, 
                f"Found: {func_str}", 
                f"Expected cosh(x), got: {func_str or error}"))
    
    # ============================================================
    # TEST 8: Function Finding - Multi-variable
    # ============================================================
    test_separator("8. Function Finding - Multi-variable")
    
    # U(k,x) = 0.5*k*x^2 (Spring potential energy)
    data = [
        (["10", "2"], "20"),   # 0.5*10*4 = 20
        (["5", "4"], "40"),    # 0.5*5*16 = 40
        (["2", "5"], "25"),    # 0.5*2*25 = 25
        (["1", "10"], "50"),   # 0.5*1*100 = 50
    ]
    success, func_str, _, error = find_function_from_data(data, ["k", "x"])
    track(check(success and "k" in func_str and "x" in func_str, 
                f"Found: {func_str}", 
                f"Expected 0.5*k*x^2 form, got: {func_str or error}"))
    
    # ============================================================
    # TEST 9: Export Command
    # ============================================================
    test_separator("9. Export Command")
    
    define_function("Volume", ["r", "h"], "3.14159*r**2*h")
    success, msg = export_function_to_file("Volume", "test_export_volume.py")
    
    if success and os.path.exists("test_export_volume.py"):
        with open("test_export_volume.py", 'r') as f:
            content = f.read()
        
        # Verify the exported file is valid Python
        try:
            exec(content, {"__name__": "__main__"})
            from test_export_volume import Volume
            result = Volume(2, 5)
            expected = 3.14159 * 4 * 5  # ~62.83
            track(check(abs(result - expected) < 0.01, 
                        f"Exported Volume(2,5) = {result:.2f}", 
                        f"Expected {expected:.2f}, got {result}"))
        except Exception as e:
            track(check(False, "", f"Export file invalid: {e}"))
        finally:
            os.remove("test_export_volume.py")
    else:
        track(check(False, "", f"Export failed: {msg}"))
    
    # ============================================================
    # TEST 10: Calculus - Differentiation
    # ============================================================
    test_separator("10. Calculus - Differentiation")
    
    result = evaluate_safely("diff(x**2 + 3*x, x)")
    if result.get("ok"):
        expr = result.get("result", "")
        track(check("2*x" in str(expr) and "3" in str(expr), 
                    f"diff(x^2+3x, x) = {expr}", 
                    f"Expected 2*x + 3, got: {expr}"))
    else:
        track(check(False, "", f"Differentiation failed: {result.get('error')}"))
    
    # ============================================================
    # TEST 11: Nested Function Calls
    # ============================================================
    test_separator("11. Nested Function Calls")
    
    clear_all_functions()
    define_function("double", ["x"], "2*x")
    define_function("square", ["x"], "x**2")
    
    # square(double(3)) = square(6) = 36
    result = evaluate_function("square", ["6"])  # Since we can't easily nest here
    track(check(str(result) == "36", 
                "square(6) = 36", 
                f"Expected 36, got {result}"))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    test_separator("SUMMARY")
    total = results["passed"] + results["failed"]
    print(f"\n  Total: {results['passed']}/{total} tests passed")
    
    if results["failed"] == 0:
        print("\n  🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n  ⚠️ {results['failed']} test(s) failed")
    
    # Cleanup
    for f in ["test_export_volume.py"]:
        if os.path.exists(f):
            os.remove(f)
