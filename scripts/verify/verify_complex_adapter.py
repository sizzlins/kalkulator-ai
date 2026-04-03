
import sys
import os
import numpy as np
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

try:
    from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
    # print(f"DEBUG: genetic_engine location: {GeneticSymbolicRegressor.__module__}, file: {GeneticSymbolicRegressor.__init__.__code__.co_filename}")
    # Use inspect for file location
    import inspect
    print(f"DEBUG: genetic_engine file: {inspect.getfile(GeneticSymbolicRegressor)}")
except ImportError:
    print("DEBUG: Could not import GeneticSymbolicRegressor directly")

from kalkulator_pkg.solver.genetic_solver_adapter import solve
import inspect
print(f"DEBUG: solve function file: {inspect.getfile(solve)}")

def verify_complex_adapter():
    """Verify genetic_solver_adapter handles complex inputs."""
    
    print("\n--- Verifying Genetic Solver Adapter Complex Input Support ---")
    
    # 1. Generate Synthetic Data for exp(sqrt(x)) with negative domain
    # Domain: [-2, -1, 1, 4]
    # x < 0 -> sqrt(x) is Imaginary -> exp(sqrt(x)) is Complex
    
    X_vals = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 4.0])
    y_vals = np.exp(np.sqrt(X_vals.astype(complex)))
    
    data_points = []
    for x, y in zip(X_vals, y_vals):
        data_points.append(((x,), y))
        
    print("Data points (y is complex):")
    for pt in data_points:
        print(f"  x={pt[0]}, y={pt[1]}")
        
    # 2. Call Adapter
    print("\nAttempting solve()...")
    # Should handle complex automatically now
    try:
        success, func_str, factored, error = solve(
            data_points, 
            param_names=["x"], 
            verbose=True,
            timeout=30,
            generations=20, # More generations for convergence
            population_size=100
        )
        
        if success:
            print(f"SUCCESS: Found '{func_str}'")
        else:
            print(f"FAILURE: {error}")
        
        print(f"  func_str={func_str}")
        print(f"  factored={factored}")
            
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_complex_adapter()
