from .dispatch import solve_single_equation
from .system import solve_system
from .inequality import solve_inequality
from .inverse import solve_inverse_function
from .utils import eval_user_expression

__all__ = [
    "solve_single_equation",
    "solve_system",
    "solve_inequality",
    "solve_inverse_function",
    "eval_user_expression",
]
