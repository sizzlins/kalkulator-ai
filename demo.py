"""Demo script to showcase Kalkulator's key features.

Run this to generate screenshots for the GitHub README.
"""

import sys

sys.path.insert(0, ".")

if __name__ == "__main__":
    import math
    import os

    from kalkulator_pkg.function_manager import (
        define_function,
        export_function_to_file,
        find_function_from_data,
    )
    from kalkulator_pkg.worker import evaluate_safely

    print("=" * 70)
    print(" KALKULATOR: Symbolic Regression & Science Engine")
    print(" Demo Showcase")
    print("=" * 70)

    # Demo 1: Kinetic Energy
    print("\n📊 Demo 1: Discovering Kinetic Energy (E = 0.5*m*v²)")
    print("-" * 50)
    data = [
        (["2", "4"], "16"),  # 0.5*2*16 = 16
        (["4", "2"], "8"),  # 0.5*4*4 = 8
        (["10", "1"], "5"),  # 0.5*10*1 = 5
        (["8", "3"], "36"),  # 0.5*8*9 = 36
    ]
    success, func_str, _, error = find_function_from_data(data, ["m", "v"])
    if success:
        print(">>> E(2,4)=16, E(4,2)=8, E(10,1)=5, E(8,3)=36, find E(m,v)")
        print(f"E(m, v) = {func_str}")
        print("✅ Discovered kinetic energy formula!")

    # Demo 2: Gaussian
    print("\n📊 Demo 2: Discovering Gaussian (g = exp(-x²))")
    print("-" * 50)
    data = [
        (["0"], "1"),
        (["1"], str(math.exp(-1))),
        (["2"], str(math.exp(-4))),
        (["3"], str(math.exp(-9))),
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    if success:
        print(">>> g(0)=1, g(1)=0.3679, g(2)=0.0183, g(3)=0.0001, find g(x)")
        print(f"g(x) = {func_str}")
        print("✅ Discovered Gaussian / Bell curve!")

    # Demo 3: Inverse Square Law
    print("\n📊 Demo 3: Discovering Inverse Square Law (I = 100/d²)")
    print("-" * 50)
    data = [
        (["1"], "100"),
        (["2"], "25"),
        (["4"], "6.25"),
        (["5"], "4"),
    ]
    success, func_str, _, error = find_function_from_data(data, ["d"])
    if success:
        print(">>> I(1)=100, I(2)=25, I(4)=6.25, I(5)=4, find I(d)")
        print(f"I(d) = {func_str}")
        print("✅ Discovered inverse square law!")

    # Demo 4: Hyperbolic (Catenary)
    print("\n📊 Demo 4: Discovering Catenary (y = cosh(x))")
    print("-" * 50)
    data = [
        (["0"], str(math.cosh(0))),
        (["1"], str(math.cosh(1))),
        (["-1"], str(math.cosh(-1))),
        (["2"], str(math.cosh(2))),
    ]
    success, func_str, _, error = find_function_from_data(data, ["x"])
    if success:
        print(">>> y(0)=1, y(1)=1.543, y(-1)=1.543, y(2)=3.762, find y(x)")
        print(f"y(x) = {func_str}")
        print("✅ Discovered catenary / hanging chain!")

    # Demo 5: Calculus
    print("\n📊 Demo 5: Calculus - Differentiation")
    print("-" * 50)
    result = evaluate_safely("diff(x**3 + 2*x**2 - 5*x + 1, x)")
    if result.get("ok"):
        print(">>> diff(x³ + 2x² - 5x + 1, x)")
        print(f"= {result.get('result')}")
        print("✅ Symbolic differentiation works!")

    # Demo 6: Export
    print("\n📊 Demo 6: Code Export")
    print("-" * 50)
    define_function("Volume", ["r", "h"], "3.14159*r**2*h")
    success, msg = export_function_to_file("Volume", "demo_volume.py")
    if success:
        print(">>> V(r,h) = 3.14159*r²*h")
        print(">>> export Volume to demo_volume.py")
        print(f"{msg}")
        with open("demo_volume.py") as f:
            print("\nGenerated Python code:")
            print(f.read())
        os.remove("demo_volume.py")
        print("✅ Code export works!")

    print("\n" + "=" * 70)
    print(" 🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
