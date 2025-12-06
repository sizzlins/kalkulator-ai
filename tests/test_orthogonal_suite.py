import subprocess
import re
import sys

def run_test(name, command_input):
    print(f"\n--- {name} ---")
    try:
        # Run CLI with input
        process = subprocess.Popen(
            [sys.executable, "-m", "kalkulator_pkg.cli"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="c:\\Users\\LOQ\\PycharmProjects\\kalkulator-ai"
        )
        stdout, stderr = process.communicate(input=command_input, timeout=10)
        
        # Extract Result
        match = re.search(r"Result: (.*)", stdout)
        if match:
            result = match.group(1).strip()
            print(f"Result: {result}")
            
            # Logic for specific checks
            if name == "Coulomb":
               # Expect q1*q2/r^2 (Inverse Square)
               res_clean = result.replace(" ", "")
               if "/r^2" in res_clean and ("q1*q2" in res_clean or "q2*q1" in res_clean):
                   print("STATUS: PASS")
                   return True
            elif name == "MechEnergy":
               # Expect m*v^2 (Interaction) beating m^2
               res_clean = result.replace(" ", "")
               if "v^2" in res_clean and "m^2" not in res_clean:
                   print("STATUS: PASS")
                   return True
            elif name == "Spacetime":
               # Expect squares (-A^2 + B^2...)
               # Ensure NO interactions like A*B^2
               res_clean = result.replace(" ", "")
               # Check for squares
               has_squares = "A^2" in res_clean and "B^2" in res_clean and "C^2" in res_clean and "D^2" in res_clean
               # Check for interactions (should NOT exist, except valid squares)
               if has_squares and "A*B" not in res_clean and "B*C" not in res_clean: 
                   print("STATUS: PASS")
                   return True
                
            print("STATUS: FAIL")
            return False
        else:
            print("No Result found.")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

# 1. Coulomb
coulomb_input = "F(1,1,1)=9000.0\nF(1,1,2)=2250.0\nF(1,1,3)=1000.0\nF(2,1,1)=18000.0\nF(2,2,2)=9000.0\nF(5,4,10)=1800.0\nF(2,5,10)=900.0\nF(1,1,10)=90.0\nF(3,3,3)=9000.0\nF(10,10,1)=900000.0\nF(1,1,0.5)=36000.0\nF(4,4,4)=9000.0\nF(2,8,4)=9000.0\nF(5,5,25)=72.0\nF(10,2,100)=18.0\nfind F(q1,q2,r)\nexit\n"
# 2. Mech Energy
mech_input = "E(1,2,0)=2.0\nE(1,0,1)=9.8\nE(2,2,1)=23.6\nE(10,10,10)=1480.0\nE(5,4,2)=138.0\nE(1,1,1)=10.3\nE(2,0,10)=196.0\nE(4,5,0)=50.0\nE(3,3,3)=116.7\nE(0.5,10,5)=49.5\nE(10,0,0)=0.0\nE(1,10,0)=50.0\nE(2,4,10)=212.0\nE(5,2,1)=59.0\nE(0.1,100,0)=500.0\nfind E(m,v,h)\nexit\n"
# 3. Spacetime
spacetime_input = "W(10,2,3,1)=-84.0\nW(2,5,1,1)=25.0\nW(1,1,1,1)=2.0\nW(5,2,2,2)=-13.0\nW(3,3,3,3)=18.0\nW(4,0,0,0)=-16.0\nW(0,4,0,0)=16.0\nW(0,0,4,0)=16.0\nW(0,0,0,4)=16.0\nW(5,5,0,0)=0.0\nW(10,1,1,1)=-97.0\nW(2,10,2,2)=100.0\nW(1,1,10,1)=100.0\nW(1,1,1,10)=100.0\nW(8,8,1,1)=2.0\nfind W(A,B,C,D)\nexit\n"

run_test("Coulomb", coulomb_input)
run_test("MechEnergy", mech_input)
run_test("Spacetime", spacetime_input)
