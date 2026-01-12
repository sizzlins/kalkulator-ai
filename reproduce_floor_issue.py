
import numpy as np
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor, GeneticConfig
from kalkulator_pkg.symbolic_regression.expression_tree import ExpressionTree
import sympy as sp

def reproduce():
    # Data from user request
    X = np.array([
        [-3.0], [-1.0], [0.0], [2.0], # Integers
        [0.2], [1.2], [2.8],          # Positive fractions
        [-0.2], [-1.2]                # Negative fractions
    ])
    
    # f(x) = floor(x) + (x - floor(x))^2
    # Calculate expected y
    y_expected = []
    for val in X.flatten():
        f_val = np.floor(val)
        rem = val - f_val
        y_val = f_val + rem**2
        y_expected.append(y_val)
    y = np.array(y_expected)
    
    print("X:", X.flatten())
    print("y:", y)
    
    # Check if seed evaluates correctly
    seed_str = "floor(x) + frac(x)**2"
    print(f"\nTesting seed: {seed_str}")
    
    try:
        # Create tree from seed
        variables = ['x']
        # Need to ensure 'frac' is understood by ExpressionTree.from_sympy or whatever parser
        # 'frac' is usually not a standard sympy function. Sympy uses 'Frac' or 'fraction'?
        # Actually sympy has 'frac' (fractional part).
        # Let's see if ExpressionTree handles it.
        
        # We manually construct tree or parsing
        # But GeneticSymbolicRegressor uses sp.sympify(seed_str)
        
        # Test sympify
        local_dict = {'x': sp.Symbol('x')}
        # frac might need to be added to local_dict if it's not in standard sympy 
        # or if we mapped it differently.
        try:
             expr = sp.sympify(seed_str, locals=local_dict)
             print("Sympify result:", expr)
        except Exception as e:
             print("Sympify failed:", e)
             # Try adding frac to locals if it's a custom function in our pkg
             # Check operators.py for 'frac' definition
             pass

        # Config with floor and frac
        config = GeneticConfig(
            population_size=100,
            generations=5,
            verbose=True,
            operators=['add', 'sub', 'mul', 'div', 'pow', 'floor', 'frac']
        )
        regressor = GeneticSymbolicRegressor(config)
        
        # Manually verify seed evaluation if possible
        # We need to access the parsers or just run fit
        
        print("\nRunning regressor.fit with explicit seed...")
        seeds = [seed_str]
        regressor.config.seeds = seeds
        
        # We run fit for a few generations
        pareto = regressor.fit(X, y, variable_names=['x'])
        
        print("\nBest solution:")
        best = pareto.get_best()
        if best:
            print("Expression:", best.expression)
            print("MSE:", best.mse)
        else:
            print("No solution found")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
