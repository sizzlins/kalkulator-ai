
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalkulator_pkg.function_manager import (
    define_function, 
    list_functions, 
    save_functions, 
    load_functions, 
    clear_functions, 
    clear_saved_functions,
    FUNCTION_STORAGE_PATH
)

def test_persistence_flow():
    print("1. Setup: Clearing everything...")
    clear_functions()
    clear_saved_functions()
    
    print("2. Define f(x)=x and save...")
    define_function("f", ["x"], "x")
    assert "f" in list_functions()
    save_functions()
    assert FUNCTION_STORAGE_PATH.exists()
    
    print("3. Clear Session (simulate 'clearfunction')...")
    clear_functions()
    assert "f" not in list_functions()
    assert FUNCTION_STORAGE_PATH.exists() # File must still exist!
    print("   [OK] Session cleared, file remains.")
    
    print("4. Load (simulate 'loadfunction')...")
    load_functions()
    assert "f" in list_functions()
    print("   [OK] Function restored from file.")
    
    print("5. Clear Save (simulate 'clearsavefunction')...")
    clear_saved_functions()
    assert not FUNCTION_STORAGE_PATH.exists()
    print("   [OK] File deleted.")
    
    print("\nSUCCESS: Logic verification passed.")

if __name__ == "__main__":
    test_persistence_flow()
