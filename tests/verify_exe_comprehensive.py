
import subprocess
import time
import os
import sys

# Assume dist/kalkulator.exe
EXE_PATH = os.path.join(os.getcwd(), "dist", "kalkulator.exe")

def run_interaction(commands, description, expect_output=None):
    """Run the exe, send commands, and verify output."""
    print(f"\n--- {description} ---")
    
    if not os.path.exists(EXE_PATH):
        print(f"Error: EXE not found at {EXE_PATH}")
        return False

    process = subprocess.Popen(
        [EXE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    
    output_log = ""
    
    try:
        # Send all commands joined by newlines
        # Add a sleep command if needed, but for now just send
        input_str = "\n".join(commands) + "\n"
        stdout, stderr = process.communicate(input=input_str, timeout=10)
        output_log = stdout + stderr
    except subprocess.TimeoutExpired:
        process.kill()
        print("TIMEOUT")
        return False
    except Exception as e:
        process.kill()
        print(f"ERROR: {e}")
        return False

    if expect_output:
        if isinstance(expect_output, list):
            # Check multiple strings
            all_found = True
            for s in expect_output:
                if s not in output_log:
                    print(f"Failure: Expected '{s}' not found.")
                    all_found = False
            if all_found:
                 print("SUCCESS: All expected outputs found.")
                 return True
            else:
                 print("Output dump:\n" + output_log)
                 return False
        else:
            if expect_output in output_log:
                print(f"SUCCESS: Found expected '{expect_output}'")
                return True
            else:
                print(f"Failure: Expected '{expect_output}' not found.")
                print("Output dump:\n" + output_log)
                return False
    else:
        print("Done (no check).")
        return True

def main():
    if not os.path.exists("dist"):
        os.makedirs("dist")
        
    # Check if exe exists
    global EXE_PATH
    if not os.path.exists(EXE_PATH):
        # Fallback to root if not in dist (sometimes verifying local)
        root_exe = os.path.join(os.getcwd(), "kalkulator.exe")
        if os.path.exists(root_exe):
            EXE_PATH = root_exe
    
    print(f"Verifying: {EXE_PATH}")

    # 1. Basic Math
    run_interaction(["2+2", "quit"], "Basic Math", "4")
    
    # 2. Function Finding
    run_interaction(
        ["f(1)=2, f(2)=4, find f(x)", "quit"], 
        "Function Finding", 
        ["f(x) = 2*x", "Function 'f' is now available"]
    )
    
    # 3. Persistence Flow
    # Clear old save
    run_interaction(["clearsavefunction", "quit"], "Persistence: Clear Old", "Saved functions cleared")
    
    # Define and Save
    run_interaction(
        ["p(x) = x^3 + 7", "savefunction", "quit"], 
        "Persistence: Save", 
        "Saved 1 function(s)"
    )
    
    # Load and Verify
    run_interaction(
        ["loadfunction", "showfunction", "quit"], 
        "Persistence: Load", 
        ["p(x)", "x**3 + 7"]
    )
    
    # Clear Save Verification
    run_interaction(
        ["clearsavefunction", "loadfunction", "quit"], 
        "Persistence: Clear Save", 
        ["Saved functions cleared", "No saved functions found"]
    )

    # 4. Export Verification
    test_file = "test_export.py"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    run_interaction(
        ["myexp(x)=x^2", "export myexp to test_export.py", "quit"], 
        "Export Function", 
        f"Function 'myexp' exported to {test_file}"
    )
    
    if os.path.exists(test_file):
        print("SUCCESS: File 'test_export.py' created on disk.")
        os.remove(test_file)
    else:
        print("FAILURE: Export file not found on disk.")

    # 5. Physics / Units (Simple check)
    # Using specific inputs to trigger physics detection
    run_interaction(
        ["E(2,4)=16, E(4,2)=8, E(10,1)=5, find E(m,v)", "quit"],
        "Physics Discovery",
        ["0.5*m*v^2"]
    )

if __name__ == "__main__":
    main()
