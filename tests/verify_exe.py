import os
import subprocess

EXE_PATH = r"c:\Users\LOQ\PycharmProjects\kalkulator-ai\kalkulator.exe"


def run_interaction(commands):
    """Run the exe, send commands, and return output."""
    process = subprocess.Popen(
        [EXE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,  # Unbuffered
    )

    try:
        # Send all commands joined by newlines
        input_str = "\n".join(commands) + "\n"
        stdout, stderr = process.communicate(input=input_str, timeout=5)
        return stdout
    except subprocess.TimeoutExpired:
        process.kill()
        return "TIMEOUT"
    except Exception as e:
        process.kill()
        return str(e)


def test_exe_persistence():
    print(f"Testing EXE at: {EXE_PATH}")
    if not os.path.exists(EXE_PATH):
        print("EXE not found!")
        return

    # Step 1: clear everything to ensure clean slate
    print("Step 1: Clear old saves...")
    run_interaction(["clearsavefunction", "quit"])

    # Step 2: Define and Save
    print("Step 2: Define and Save...")
    run_interaction(["g(x) = x^3 + 10", "savefunction", "quit"])

    # Step 3: Load and Verify
    print("Step 3: Load and Verify...")
    out3 = run_interaction(["loadfunction", "showfunction", "quit"])

    print("-" * 20)
    print("OUTPUT CHECK:")
    if "g(x)" in out3 and "x**3 + 10" in out3:
        print("SUCCESS: Function 'g(x)' was successfully saved and loaded.")
    else:
        print("FAILURE: Function 'g(x)' was NOT found after loading.")
        print("Output from Step 3:")
        print(out3)

    # Step 4: Verify Clear Save
    print("Step 4: Verify Clear Save logic...")
    out4 = run_interaction(["clearsavefunction", "loadfunction", "quit"])

    if "No saved functions found" in out4:
        print("SUCCESS: clearsavefunction worked.")
    else:
        print("FAILURE: clearsavefunction might have failed.")
        print(out4)


if __name__ == "__main__":
    test_exe_persistence()
