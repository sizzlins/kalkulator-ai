from sympy.ntheory.modular import crt, solve_congruence
from kalkulator_pkg.solver.modular import solve_system_of_congruences
import time

def test_non_coprime():
    # x = 1 mod 4
    # x = 3 mod 6
    # Solution: x = 9 mod 12
    congruences = [(1, 4), (3, 6)]
    
    print("Testing Non-Coprime Moduli (4, 6)...")
    
    # 1. Custom Implementation
    try:
        start = time.perf_counter()
        k1, m1 = solve_system_of_congruences(congruences)
        res1 = f"x = {k1} mod {m1}"
        t1 = time.perf_counter() - start
        print(f"[Custom] Result: {res1} (Time: {t1:.6f}s)")
    except Exception as e:
        print(f"[Custom] Failed: {e}")

    # 2. SymPy crt
    try:
        remainders = [1, 3]
        moduli = [4, 6]
        start = time.perf_counter()
        res2 = crt(moduli, remainders)
        t2 = time.perf_counter() - start
        print(f"[SymPy crt] Result: {res2} (Time: {t2:.6f}s)")
    except Exception as e:
        print(f"[SymPy crt] Failed: {e}")

    # 3. SymPy solve_congruence
    try:
        # solve_congruence takes tuples (a, n)
        start = time.perf_counter()
        res3 = solve_congruence(*congruences)
        t3 = time.perf_counter() - start
        print(f"[SymPy solve_congruence] Result: {res3} (Time: {t3:.6f}s)")
    except Exception as e:
        print(f"[SymPy solve_congruence] Failed: {e}")

if __name__ == "__main__":
    test_non_coprime()
