
try:
    from kalkulator_pkg.symbolic_regression.operators import crossover
    print(f"Recursion check: {crossover}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
    print("GeneticSymbolicRegressor imported successfully")
except Exception as e:
    print(f"GeneticEngine Error: {e}")
