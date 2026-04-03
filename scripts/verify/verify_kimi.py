import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from kalkulator_pkg.ai_utils import get_kimi

def test_kimi():
    print("Testing Kimi Integration...")
    kimi = get_kimi()
    
    exe = kimi._find_executable()
    print(f"Executable path: {exe}")
    
    if not kimi.is_available():
        print("FAIL: Kimi not available")
        return
        
    print("Sending query 'hello'...")
    response = kimi.query("hello")
    print(f"Response length: {len(response)}")
    print(f"Response preview: {response[:100]}...")
    
    if len(response) > 0:
        print("PASS: Received response")
    else:
        print("FAIL: Empty response")

if __name__ == "__main__":
    test_kimi()
