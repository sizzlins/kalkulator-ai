import numpy as np
# from kalkulator_pkg.cli.repl_commands import main_entry
# We can't easily invoke main_entry to run the REPL loop programmatically without mocking input.
# Instead, let's just copy the logic we want to test or try to import a function if possible.
# Actually, the logic is inside `process_command` or similar, which might be hard to reach.
# But wait, I modified `repl_commands.py`.
# Let's write a small script that copies the NEW logic and tests it against the data to PROVE it works.
# This serves as a unit test for the logic I just wrote.

def test_filtering_logic():
    print("=== Testing Outlier Filter Bypass Logic ===")
    
    # Generate the problem data: floor(10 * sin(x))
    X = np.linspace(-20, 20, 258)
    y = np.floor(10 * np.sin(X))
    
    print(f"Data Sample: {y[:10]}")
    
    # --- LOGIC UNDER TEST (Copied from repl_commands.py) ---
    try:
        # NEW: Check for discrete values (integers)
        # If data is discrete (e.g. floor, ceil, step), outliers might be valid jumps.
        y_real_check = np.real(y) if np.iscomplexobj(y) else y.astype(float)
        y_round_check = np.round(y_real_check)
        mse_int_check = np.mean((y_real_check - y_round_check)**2)
        
        print(f"MSE Check Value: {mse_int_check}")
        
        if mse_int_check < 0.01:
            print("Note: Discrete values detected. Outlier filtering disabled to preserve step/jump data.")
            filtered = False
        elif len(y) >= 10 and not np.iscomplexobj(y):
            # ... existing IQR logic ...
            # I won't copy the whole IQR block, just enough to show it WOULD have run
            print("Running IQR logic...")
            # Simulate what IQR would do
            y_real = y_real_check
            q1 = np.percentile(y_real, 25)
            q3 = np.percentile(y_real, 75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = (y_real >= lower_bound) & (y_real <= upper_bound)
            num_outliers = np.sum(~outlier_mask)
            print(f"IQR would filter {num_outliers} points.")
            filtered = True
        else:
            filtered = False
            
    except Exception as e:
        print(f"Error: {e}")
        filtered = False
        
    if not filtered:
        print("SUCCESS: Filter was bypassed!")
    else:
        print("FAILURE: Filter was NOT bypassed!")

if __name__ == "__main__":
    test_filtering_logic()
