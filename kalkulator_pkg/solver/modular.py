from __future__ import annotations

import sympy as sp

try:
    from ..logging_config import get_logger

    logger = get_logger("solver.modular")
except ImportError:
    import logging

    logger = logging.getLogger("solver.modular")


def _solve_modulo_equation(equation: sp.Eq, variable: sp.Symbol) -> list[sp.Basic]:
    """Solve a modulo equation of the form Mod(x, n) = k or x % n = k.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of solutions (parametric form if possible)
    """
    try:
        # Check if equation is of the form Mod(x, n) = k or x % n = k
        lhs = equation.lhs
        rhs = equation.rhs

        # Check if LHS is Mod(variable, n)
        if isinstance(lhs, sp.Mod):
            # Mod(x, n) = k
            if lhs.args[0] == variable:
                n = lhs.args[1]
                k = rhs
                # Solution: x = k + n*t where t is an integer
                t = sp.symbols("t", integer=True)
                solution = k + n * t
                return [solution]

        # Try SymPy's solve for modulo equations
        try:
            solutions = sp.solve(equation, variable, domain=sp.Integers)
            if solutions:
                if isinstance(solutions, dict):
                    sol = solutions.get(variable)
                    if sol is not None:
                        return [sol]
                elif isinstance(solutions, (list, tuple)) and solutions:
                    return list(solutions)
                elif solutions:
                    return [solutions]
        except (NotImplementedError, ValueError, TypeError):
            pass

        # If no symbolic solution, try numeric approach
        # For Mod(x, n) = k, solutions are x = k + n*t for integer t
        # Return parametric solution
        if isinstance(lhs, sp.Mod) and lhs.args[0] == variable:
            n = lhs.args[1]
            k = rhs
            try:
                # Try to get numeric values
                n_val = float(sp.N(n))
                k_val = float(sp.N(k))
                if n_val > 0:
                    t = sp.symbols("t", integer=True)
                    solution = k_val + n_val * t
                    return [solution]
            except (ValueError, TypeError):
                pass

        return []
    except (NotImplementedError, ValueError, TypeError, AttributeError):
        return []


def solve_system_of_congruences(
    congruences: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Solve a system of congruences using the Chinese Remainder Theorem.

    Solves the system:
        x ≡ a₁ (mod n₁)
        x ≡ a₂ (mod n₂)
        ...
        x ≡ aₖ (mod nₖ)

    Args:
        congruences: List of (a, n) tuples representing x ≡ a (mod n)

    Returns:
        Tuple (k, m) representing the solution x ≡ k (mod m), or None if no solution exists
    """
    if not congruences:
        return None

    try:
        # USE STANDARD LIBRARY: sympy.ntheory.modular.solve_congruence
        # This handles non-coprime moduli correctly and is maintained by SymPy.
        from sympy.ntheory.modular import solve_congruence

        # solve_congruence takes arguments as tuples (rem, mod)
        # Our input is (rem, mod) tuples, so we unpack them
        result = solve_congruence(*congruences)

        if result is None:
            return None

        return result

        # LEGACY MANUAL IMPLEMENTATION (COMMENTED OUT FOR PRESERVATION)
        # Use SymPy's crt function if available
        # if hasattr(sp, "crt"):
        #     remainders = [a for a, n in congruences]
        #     moduli = [n for a, n in congruences]
        #     try:
        #         result = sp.crt(moduli, remainders)
        #         if result is None:
        #             return None
        #         # Calculate the modulus (LCM of all moduli)
        #         from math import lcm  # noqa: F811
        #
        #         m = lcm(*moduli)
        #         # Normalize the remainder to be in [0, m)
        #         k = result % m
        #         return (k, m)
        #     except (ValueError, TypeError, NotImplementedError):
        #         pass

        # Manual implementation of CRT for two congruences at a time
        # Start with first congruence
        # k, m = congruences[0]
        # k = k % m  # Normalize
        #
        # # Combine with each subsequent congruence
        # for a, n in congruences[1:]:
        #     a = a % n  # Normalize
        #
        #     # Check if moduli are coprime (required for CRT)
        #     # If not coprime, we need to check consistency
        #     from math import gcd, lcm  # noqa: F811, F401, F402
        #
        #     g = gcd(m, n)
        #
        #     # Check consistency: k ≡ a (mod g)
        #     if (k % g) != (a % g):
        #         # System is inconsistent
        #         return None
        #
        #     # Combine congruences x ≡ k (mod m) and x ≡ a (mod n)
        #     # We need to find x such that:
        #     #   x = k + m*t₁  (for some integer t₁)
        #     #   x = a + n*t₂  (for some integer t₂)
        #     # This means: k + m*t₁ ≡ a (mod n)
        #     # Rearranging: m*t₁ ≡ a - k (mod n)
        #
        #     # Solve m*t ≡ (a - k) (mod n)
        #     # This requires gcd(m, n) to divide (a - k), which we already checked
        #
        #     # Use extended Euclidean algorithm to find t
        #     # m*t ≡ (a - k) (mod n)
        #     m_mod = m % n
        #     target = (a - k) % n
        #
        #     if g == 1:
        #         # Moduli are coprime - use multiplicative inverse
        #         # Find t such that m_mod * t ≡ target (mod n)
        #         try:
        #             m_inv = pow(int(m_mod), -1, int(n))  # Modular inverse
        #             t = (target * m_inv) % n
        #         except ValueError:
        #             # Fallback: try SymPy
        #             t_sym = sp.symbols("t", integer=True)
        #             eq = sp.Eq(m_mod * t_sym, target)
        #             try:
        #                 sol = sp.solve(eq, t_sym, domain=sp.Integers)
        #                 if sol:
        #                     if isinstance(sol, dict):
        #                         t = int(sol[t_sym]) % n
        #                     elif isinstance(sol, list) and sol:
        #                         t = int(sol[0]) % n
        #                     else:
        #                         return None
        #                 else:
        #                     return None
        #             except (ValueError, TypeError, NotImplementedError):
        #                 return None
        #     else:
        #         # Moduli are not coprime, but system is consistent
        #         # Divide by gcd: (m/g) * t ≡ (a-k)/g (mod n/g)
        #         m_div = m_mod // g
        #         n_div = n // g
        #         target_div = target // g
        #
        #         try:
        #             m_inv = pow(int(m_div), -1, int(n_div))
        #             t = (target_div * m_inv) % n_div
        #         except ValueError:
        #             # Fallback
        #             t_sym = sp.symbols("t", integer=True)
        #             eq = sp.Eq(m_div * t_sym, target_div)
        #             try:
        #                 sol = sp.solve(eq, t_sym, domain=sp.Integers)
        #                 if sol:
        #                     if isinstance(sol, dict):
        #                         t = int(sol[t_sym]) % n_div
        #                     elif isinstance(sol, list) and sol:
        #                         t = int(sol[0]) % n_div
        #                     else:
        #                         return None
        #                 else:
        #                     return None
        #             except (ValueError, TypeError, NotImplementedError):
        #                 return None
        #
        #     # Update solution: x = k + m*t
        #     k = k + m * t
        #     m = lcm(m, n)
        #     k = k % m  # Normalize
        #
        # return (k, m)

    except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
        logger.debug(f"Error solving system of congruences: {e}")
        return None
        #     moduli = [n for a, n in congruences]
        #     try:
        #         result = sp.crt(moduli, remainders)
        #         if result is None:
        #             return None
        #         # Calculate the modulus (LCM of all moduli)
        #         from math import lcm  # noqa: F811
        #
        #         m = lcm(*moduli)
        #         # Normalize the remainder to be in [0, m)
        #         k = result % m
        #         return (k, m)
        #     except (ValueError, TypeError, NotImplementedError):
        #         pass

        # USE STANDARD LIBRARY: sympy.ntheory.modular.solve_congruence
        # This handles non-coprime moduli correctly and is maintained by SymPy.
        from sympy.ntheory.modular import solve_congruence

        # solve_congruence takes arguments as tuples (rem, mod)
        # Our input is (rem, mod) tuples, so we unpack them
        result = solve_congruence(*congruences)

        if result is None:
            return None

        return result

        # LEGACY MANUAL IMPLEMENTATION (COMMENTED OUT FOR PRESERVATION)
        # Manual implementation of CRT for two congruences at a time
        # Start with first congruence
        # k, m = congruences[0]
        # k = k % m  # Normalize
        #
        # # Combine with each subsequent congruence
        # for a, n in congruences[1:]:
        #     a = a % n  # Normalize
        #
        #     # Check if moduli are coprime (required for CRT)
        #     # If not coprime, we need to check consistency
        #     from math import gcd, lcm  # noqa: F811, F401, F402
        #
        #     g = gcd(m, n)
        #
        #     # Check consistency: k ≡ a (mod g)
        #     if (k % g) != (a % g):
        #         # System is inconsistent
        #         return None
        #
        #     # Combine congruences x ≡ k (mod m) and x ≡ a (mod n)
        #     # We need to find x such that:
        #     #   x = k + m*t₁  (for some integer t₁)
        #     #   x = a + n*t₂  (for some integer t₂)
        #     # This means: k + m*t₁ ≡ a (mod n)
        #     # Rearranging: m*t₁ ≡ a - k (mod n)
        #
        #     # Solve m*t ≡ (a - k) (mod n)
        #     # This requires gcd(m, n) to divide (a - k), which we already checked
        #
        #     # Use extended Euclidean algorithm to find t
        #     # m*t ≡ (a - k) (mod n)
        #     m_mod = m % n
        #     target = (a - k) % n
        #
        #     if g == 1:
        #         # Moduli are coprime - use multiplicative inverse
        #         # Find t such that m_mod * t ≡ target (mod n)
        #         try:
        #             m_inv = pow(int(m_mod), -1, int(n))  # Modular inverse
        #             t = (target * m_inv) % n
        #         except ValueError:
        #             # Fallback: try SymPy
        #             t_sym = sp.symbols("t", integer=True)
        #             eq = sp.Eq(m_mod * t_sym, target)
        #             try:
        #                 sol = sp.solve(eq, t_sym, domain=sp.Integers)
        #                 if sol:
        #                     if isinstance(sol, dict):
        #                         t = int(sol[t_sym]) % n
        #                     elif isinstance(sol, list) and sol:
        #                         t = int(sol[0]) % n
        #                     else:
        #                         return None
        #                 else:
        #                     return None
        #             except (ValueError, TypeError, NotImplementedError):
        #                 return None
        #     else:
        #         # Moduli are not coprime, but system is consistent
        #         # Divide by gcd: (m/g) * t ≡ (a-k)/g (mod n/g)
        #         m_div = m_mod // g
        #         n_div = n // g
        #         target_div = target // g
        #
        #         try:
        #             m_inv = pow(int(m_div), -1, int(n_div))
        #             t = (target_div * m_inv) % n_div
        #         except ValueError:
        #             # Fallback
        #             t_sym = sp.symbols("t", integer=True)
        #             eq = sp.Eq(m_div * t_sym, target_div)
        #             try:
        #                 sol = sp.solve(eq, t_sym, domain=sp.Integers)
        #                 if sol:
        #                     if isinstance(sol, dict):
        #                         t = int(sol[t_sym]) % n_div
        #                     elif isinstance(sol, list) and sol:
        #                         t = int(sol[0]) % n_div
        #                     else:
        #                         return None
        #                 else:
        #                     return None
        #             except (ValueError, TypeError, NotImplementedError):
        #                 return None
        #
        #     # Update solution: x = k + m*t
        #     k = k + m * t
        #     m = lcm(m, n)
        #     k = k % m  # Normalize
        #
        # return (k, m)

    except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
        logger.debug(f"Error solving system of congruences: {e}")
        return None
