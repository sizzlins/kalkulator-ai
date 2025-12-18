from ..types import EvalResult
from ..worker import evaluate_safely


def eval_user_expression(expr: str) -> EvalResult:
    data = evaluate_safely(expr)
    if not data.get("ok"):
        return EvalResult(ok=False, error=data.get("error") or "Unknown error")
    return EvalResult(
        ok=True,
        result=data.get("result"),
        approx=data.get("approx"),
        free_symbols=data.get("free_symbols"),
    )
