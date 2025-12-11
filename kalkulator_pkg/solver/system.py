import json
import sympy as sp
from typing import Any
from ..config import VAR_NAME_RE
from ..worker import evaluate_safely, _worker_solve_cached
from ..parser import parse_preprocessed, split_top_level_commas

try:
    from ..logging_config import get_logger
    logger = get_logger("solver.system")
except ImportError:
    import logging
    logger = logging.getLogger("solver.system")

ZERO_TOL = 1e-12

def solve_system(
    raw_no_find: str,
    find_token: str | None,
    allowed_functions: frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Solve a system of equations.

    Args:
        raw_no_find: Comma-separated equation strings (e.g., "x+y=3, x-y=1")
        find_token: Optional variable to find in solutions

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: Result type ("system", "system_var")
            - solutions: List of solution dictionaries (for system)
            - exact: List of exact solutions (for system_var)
            - approx: List of approximate solutions (for system_var)
            - error: Error message if ok is False
    """
    parts = [p.strip() for p in split_top_level_commas(raw_no_find) if p.strip()]
    eqs_serialized = []
    assignments = {}
    for p in parts:
        if "=" not in p:
            continue
        lhs, rhs = p.split("=", 1)
        lhs_s = lhs.strip()
        rhs_s = rhs.strip()
        lhs_eval = evaluate_safely(lhs_s, allowed_functions=allowed_functions)
        if not lhs_eval.get("ok"):
            return {
                "ok": False,
                "error": f"LHS parse error: {lhs_eval.get('error')}",
                "error_code": lhs_eval.get("error_code", "PARSE_ERROR"),
            }
        rhs_eval = evaluate_safely(rhs_s, allowed_functions=allowed_functions)
        if not rhs_eval.get("ok"):
            return {
                "ok": False,
                "error": f"RHS parse error: {rhs_eval.get('error')}",
                "error_code": rhs_eval.get("error_code", "PARSE_ERROR"),
            }
        if VAR_NAME_RE.match(lhs_s):
            assignments[lhs_s] = {
                "result": rhs_eval.get("result"),
                "approx": rhs_eval.get("approx"),
            }
        eqs_serialized.append(
            {"lhs": lhs_eval.get("result"), "rhs": rhs_eval.get("result")}
        )
    if not eqs_serialized:
        return {
            "ok": False,
            "error": "No equations parsed.",
            "error_code": "NO_EQUATIONS",
        }
    if find_token and len(eqs_serialized) == 1:
        pair = eqs_serialized[0]
        lhs_s = pair.get("lhs")
        rhs_s = pair.get("rhs")
        try:
            lhs_expr = parse_preprocessed(lhs_s)
            rhs_expr = parse_preprocessed(rhs_s)
            equation = sp.Eq(lhs_expr, rhs_expr)
            sym = sp.symbols(find_token)
            if sym in equation.free_symbols:
                try:
                    sols = sp.solve(equation, sym)
                except (NotImplementedError, ValueError, TypeError) as e:
                    logger.warning(f"Error solving inequality: {e}", exc_info=True)
                    sols = []
                if sols:
                    exacts = (
                        [str(s) for s in sols]
                        if isinstance(sols, (list, tuple))
                        else [str(sols)]
                    )
                    approx = []
                    for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                        try:
                            approx.append(str(sp.N(s)))
                        except (ValueError, TypeError, OverflowError, ArithmeticError):
                            # Expected for some symbolic solutions
                            approx.append(None)
                    return {
                        "ok": True,
                        "type": "system_var",
                        "exact": exacts,
                        "approx": approx,
                    }
        except (ValueError, TypeError, AttributeError):
            # Expected for some expressions that can't be processed
            pass
    if find_token:
        defining = None
        for pair in eqs_serialized:
            if pair.get("rhs") == find_token:
                defining = ("lhs", pair.get("lhs"))
                break
            if pair.get("lhs") == find_token:
                defining = ("rhs", pair.get("rhs"))
                break
        if defining:
            side, expr_str = defining
            try:
                expr_sym = parse_preprocessed(expr_str)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error extracting expression symbol: {e}", exc_info=True)
                expr_sym = None
            if expr_sym is not None:
                subs_map = {}
                for var, info in assignments.items():
                    if info.get("approx") is not None:
                        try:
                            subs_map[sp.symbols(var)] = sp.sympify(info.get("approx"))
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                f"Error in inequality solving: {e}", exc_info=True
                            )
                            try:
                                subs_map[sp.symbols(var)] = parse_preprocessed(
                                    info.get("result")
                                )
                            except (ParseError, ValidationError, ValueError, TypeError):
                                # Expected parsing errors
                                pass
                    else:
                        try:
                            subs_map[sp.symbols(var)] = parse_preprocessed(
                                info.get("result")
                            )
                        except (ParseError, ValidationError, ValueError, TypeError):
                            # Expected parsing errors
                            pass
                if subs_map:
                    try:
                        value = expr_sym.subs(subs_map)
                        try:
                            approx_obj = sp.N(value)
                            if (
                                abs(sp.re(approx_obj)) < ZERO_TOL
                                and abs(sp.im(approx_obj)) < ZERO_TOL
                            ):
                                return {
                                    "ok": True,
                                    "type": "system_var",
                                    "exact": ["0"],
                                    "approx": ["0"],
                                }
                            approx_val = str(approx_obj)
                        except (ValueError, TypeError, OverflowError, ArithmeticError):
                            # Expected for some symbolic expressions
                            approx_val = None
                        return {
                            "ok": True,
                            "type": "system_var",
                            "exact": [str(value)],
                            "approx": [approx_val],
                        }
                    except (ValueError, TypeError, AttributeError):
                        # Expected for some expressions
                        pass
    payload = {"equations": eqs_serialized, "find": find_token}
    try:
        stdout_text = _worker_solve_cached(json.dumps(payload))
    except (TimeoutError, ValueError, TypeError) as e:
        logger.warning(f"Error in worker-based solving: {e}", exc_info=True)
        return {
            "ok": False,
            "error": "Solving timed out (worker).",
            "error_code": "TIMEOUT",
        }
    try:
        data_untyped = json.loads(stdout_text)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Invalid JSON from worker-solve: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Invalid worker-solve output: {e}.",
            "error_code": "INVALID_OUTPUT",
        }

    if not isinstance(data_untyped, dict):
        logger.warning("Worker-solve returned non-object JSON", exc_info=True)
        return {
            "ok": False,
            "error": "Invalid worker-solve output: expected JSON object.",
            "error_code": "INVALID_OUTPUT",
        }

    data: dict[str, Any] = data_untyped
    if not data.get("ok"):
        return data
    sols_list = data.get("solutions", [])
    if not find_token:
        return {"ok": True, "type": "system", "solutions": sols_list}

    found_vals: list[str] = []
    for sol_dict in sols_list:
        v = sol_dict.get(find_token)
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        found_vals.append(v)

    if not found_vals:
        return {
            "ok": False,
            "error": f"No solution found for variable {find_token}.",
            "error_code": "NO_SOLUTION",
        }

    approx_vals: list[str | None] = []
    for vstr in found_vals:
        try:
            val_sym = sp.sympify(vstr)
            approx_val = str(sp.N(val_sym))
            approx_vals.append(approx_val)
        except (ValueError, TypeError, OverflowError, ArithmeticError):
            # Expected for some symbolic solutions
            approx_vals.append(None)
    return {
        "ok": True,
        "type": "system_var",
        "exact": found_vals,
        "approx": approx_vals,
    }
