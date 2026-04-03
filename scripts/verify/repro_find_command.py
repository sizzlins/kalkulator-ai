
import sys
import os

# Ensure package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.cli.handlers.discovery import handle_find_command_raw
from kalkulator_pkg.cli.context import Context

def test_find_command():
    ctx = Context()
    
    # Simulate User Input (pasted from callr output)
    # Copied partially from user prompt
    raw_input = """f(-2.9) = 0.8247700809218724, f(-16) = 0.8164000595236316, f(2.4) = 0.572036349286564, f(-sin(1)) = 0.4228940785616858, f(4.2) = -0.20406515688401095, f(-0.15) = 0.8352574314191994, find f"""
    
    print("Testing handle_find_command_raw with raw input...")
    try:
        result = handle_find_command_raw(raw_input, ctx)
        print(f"Result: {result}")
        if result:
            print("Successfully handled.")
        else:
            print("Failed to handle (returned False).")
    except Exception as e:
        print(f"Crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_find_command()
