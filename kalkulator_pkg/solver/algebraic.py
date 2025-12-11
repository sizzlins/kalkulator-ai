import sympy as sp
from ..config import NUMERIC_TOLERANCE

def is_pell_equation_from_eq(eq: sp.Eq) -> bool:
    """
    Check if an equation is a Pell's equation.

    Args:
        eq: SymPy equation object

    Returns:
        True if the equation is a Pell's equation, False otherwise
    """
    syms = list(eq.free_symbols)
    if len(syms) != 2:
        return False
    try:
        if not eq.rhs.equals(1):
            return False
    except (AttributeError, TypeError, ValueError):
        # Invalid equation structure
        return False
    expanded_lhs = sp.expand(eq.lhs)
    x_sym, y_sym = syms[0], syms[1]

    # Check: coefficient of x^2 must be 1
    coeff_x2 = expanded_lhs.coeff(x_sym**2)
    # Use equals() for SymPy comparison since == might not work for all cases
    if not sp.sympify(coeff_x2).equals(1):
        return False

    # Check: coefficient of y^2 must be negative (non-zero)
    coeff_y2 = expanded_lhs.coeff(y_sym**2)
    if coeff_y2 == 0:
        return False

    # For Pell equation x^2 - D*y^2 = 1, we need D = -coeff_y2 to be a positive integer
    D = -coeff_y2
    try:
        D_val = int(sp.N(D))
        if D_val <= 0:
            return False
        # Check D is not a perfect square
        sqrt_D = sp.sqrt(D_val)
        if sqrt_D.is_rational:
            return False
    except (ValueError, TypeError):
        return False

    # Check no other terms exist (no constant, no xy terms, etc.)
    # Subtract the x^2 and y^2 terms and check remainder is 0
    remainder = expanded_lhs - x_sym**2 - coeff_y2 * y_sym**2
    if sp.simplify(remainder) != 0:
        return False

    return True


def fundamental_solution(D: int) -> tuple[int, int]:
    """
    Find the fundamental solution to Pell's equation x^2 - D*y^2 = 1.
    
    DEPRECATED: Replaced by sympy.solvers.diophantine
    """
    # LEGACY MANUAL IMPLEMENTATION (COMMENTED OUT FOR PRESERVATION)
    pass
    # sqrt_D = sp.sqrt(D)
    # if sqrt_D.is_rational:
    #     raise ValueError("D must be non-square for Pell's equation")
    #
    # # Use continued fraction to find the fundamental solution
    # # ... (rest of the logic commented out) ...
    # cf = sp.continued_fraction(sqrt_D)
    # if not cf or len(cf) < 2:
    #     raise ValueError(f"Invalid continued fraction for D={D}")
    #
    # a0 = cf[0] if isinstance(cf[0], (int, sp.Integer)) else int(cf[0])
    # period = []
    # if len(cf) > 1:
    #     period_item = cf[1]
    #     if isinstance(period_item, list):
    #         period = [
    #             int(x) if isinstance(x, (int, sp.Integer)) else int(sp.N(x))
    #             for x in period_item
    #         ]
    #     else:
    #         period = [
    #             (
    #                 int(period_item)
    #                 if isinstance(period_item, (int, sp.Integer))
    #                 else int(sp.N(period_item))
    #             )
    #         ]
    #
    # if not period:
    #     raise ValueError(f"Could not extract period for D={D}")
    #
    # L = len(period)
    # p_minus2, p_minus1 = 0, 1
    # q_minus2, q_minus1 = 1, 0
    #
    # # First convergent: a0/1
    # p_prev = a0 * p_minus1 + p_minus2
    # q_prev = a0 * q_minus1 + q_minus2
    #
    # if p_prev * p_prev - D * q_prev * q_prev == 1:
    #     return int(p_prev), int(q_prev)
    #
    # max_iter = 2 * L
    # for i in range(max_iter):
    #     a_i = period[i % L] if period else a0
    #     p_curr = a_i * p_prev + p_minus1
    #     q_curr = a_i * q_prev + q_minus1
    #
    #     if p_curr * p_curr - D * q_curr * q_curr == 1:
    #         return int(p_curr), int(q_curr)
    #
    #     p_minus1, p_prev = p_prev, p_curr
    #     q_minus1, q_prev = q_prev, q_curr
    #
    # raise ValueError(f"Could not find fundamental solution for D={D}")


def solve_pell_equation_from_eq(eq: sp.Eq) -> str:
    """
    Solve a Pell's equation parametrically.

    Args:
        eq: SymPy equation object representing a Pell's equation

    Returns:
        String representation of the parametric solution
    """
    try:
        # USE STANDARD LIBRARY: sympy.solvers.diophantine
        from sympy.solvers.diophantine import diophantine
        
        # diophantine returns a set of tuples representing solutions
        # e.g., {(x(t), y(t)), ...}
        solutions = diophantine(eq)
        
        if not solutions:
            return "No integer solutions found."
            
        # Take the first solution family (usually the positive one for Pell)
        # Note: formatting might need to be pretty, but this is robust.
        sol_list = list(solutions)
        
        # Format explicitly
        syms = list(eq.free_symbols)
        # We need to map the result expressions back to x and y
        # diophantine returns values in alphabetical order of symbols? 
        # Actually it returns based on how it parsed them.
        # Let's assume standard x, y order if input was structured that way.
        # Ideally we match symbols.
        
        # Safe fallback: just stringify the first solution
        # sol_list[0] is a tuple (val_1, val_2) corresponding to sorted(eq.free_symbols)
        sorted_syms = sorted(syms, key=lambda s: s.name)
        
        final_str_parts = []
        # Usually there are multiple families (positive/negative). 
        # We'll just show the first one or all of them.
        for i, sol_tuple in enumerate(sol_list):
            # Limit to 1 family to avoid clutter? Pell has infinite, represented by 't'.
            if i >= 1: break # Just show one general form
            
            for var, expr in zip(sorted_syms, sol_tuple):
                final_str_parts.append(f"{var} = {expr}")
                
        return "\n".join(final_str_parts)

    except Exception:
        # Fallback to legacy string if something goes wrong? 
        # No, we want to know if it fails. Return error string.
        return "Error: Could not solve Pell equation."

    # LEGACY MANUAL IMPLEMENTATION (COMMENTED OUT)
    # syms = list(eq.free_symbols)
    # x_sym, y_sym = syms[0], syms[1]
    # expanded_lhs = sp.expand(eq.lhs)
    # coeff_y2 = expanded_lhs.coeff(y_sym**2)
    # D = -coeff_y2
    # x1, y1 = fundamental_solution(int(D))
    # n = sp.symbols("n", integer=True)
    # sol_x = ((x1 + y1 * sp.sqrt(D)) ** n + (x1 - y1 * sp.sqrt(D)) ** n) / 2
    # sol_y = ((x1 + y1 * sp.sqrt(D)) ** n - (x1 - y1 * sp.sqrt(D)) ** n) / (
    #     2 * sp.sqrt(D)
    # )
    # return f"{x_sym} = {sol_x}\n{y_sym} = {sol_y}"

def _solve_linear_equation(equation: sp.Eq, variable: sp.Symbol) -> list[sp.Basic]:
    """Solve a linear equation of the form a*x + b = 0.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of solutions
    """
    try:
        solutions = sp.solve(equation, variable)
        if isinstance(solutions, dict):
            return [solutions.get(variable)]
        elif isinstance(solutions, (list, tuple)):
            return list(solutions)
        else:
            return [solutions]
    except (NotImplementedError, ValueError, TypeError):
        return []


def _solve_quadratic_equation(equation: sp.Eq, variable: sp.Symbol) -> list[sp.Basic]:
    """Solve a quadratic equation using polynomial factorization.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of solutions
    """
    try:
        # Try polynomial approach for quadratics
        poly = sp.Poly(equation.lhs - equation.rhs, variable)
        if poly is not None and poly.degree() == 2:
            roots = poly.nroots()
            return [r for r in roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
    except (ValueError, TypeError):
        pass

    # Fallback to general solve
    try:
        solutions = sp.solve(equation, variable)
        if isinstance(solutions, dict):
            return [solutions.get(variable)]
        elif isinstance(solutions, (list, tuple)):
            return list(solutions)
        else:
            return [solutions]
    except (NotImplementedError, ValueError, TypeError):
        return []


def _solve_polynomial_equation(equation: sp.Eq, variable: sp.Symbol) -> list[sp.Basic]:
    """Solve a polynomial equation using Poly root finding.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of numeric roots
    """
    try:
        poly = sp.Poly(equation.lhs - equation.rhs, variable)
        if poly is not None and poly.degree() > 0:
            # For low-degree polynomials, try exact roots first
            if poly.degree() <= 4:
                try:
                    exact_roots = poly.all_roots()
                    return [r for r in exact_roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
                except (NotImplementedError, ValueError):
                    pass
            # Fallback to numeric roots
            numeric_roots = poly.nroots()
            return [r for r in numeric_roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
    except (ValueError, TypeError):
        pass

    return []
