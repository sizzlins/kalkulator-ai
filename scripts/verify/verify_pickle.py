
import sys
import pickle
import os

# Ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

try:
    from kalkulator_pkg.symbolic_regression.population import Population
    from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig
    
    print("[INFO] Imported Population class successfully.")
    
    # Create an instance
    config = GeneticConfig()
    pop = Population(size=10, variable_names=['x'], config=config, random_state=42)
    
    print("[INFO] Created Population instance.")
    
    # Try to pickle
    dumped = pickle.dumps(pop)
    print("[INFO] Pickled Population instance successfully.")
    
    # Try to unpickle
    loaded = pickle.loads(dumped)
    print("[INFO] Unpickled Population instance successfully.")
    
    print("\n[SUCCESS] Serialization Test Passed!")

except AttributeError as e:
    print(f"\n[FAIL] Serialization Failed: {e}")
    if "Can't pickle local object" in str(e):
        print("Reason: The class is still defined locally inside a function.")
except Exception as e:
    print(f"\n[FAIL] Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
