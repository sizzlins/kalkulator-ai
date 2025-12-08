import re
import sympy as sp
from typing import Any
from ..worker import evaluate_safely
from ..parser import parse_preprocessed
from ..types import ParseError, ValidationError

try:
    from ..logging_config import get_logger
    logger = get_logger("solver.inequality")
except ImportError:
    import logging
    logger = logging.getLogger("solver.inequality")

def _parse_relational_fallback(rel_str: str) -> sp.Basic:
    """Parse a relational expression fallback method.

    Args:
        rel_str: String containing relational operator

    Returns:
        SymPy expression (if single expression) or tuple of parsed parts
        Raises ValueError if parsing fails
    """
    parts = re.split(r"(<=|>=|<|>)", rel_str)
    if len(parts) == 1:
        res = evaluate_safely(rel_str)
        if not res.get("ok"):
            raise ValueError(res.get("error"))
        return parse_preprocessed(res["result"])
    expr_parts = parts[::2]
    ops = parts[1::2]
    parsed_parts = []
    for p in expr_parts:
        p_strip = p.strip()
        if not p_strip:
            continue
        res = evaluate_safely(p_strip)
        if not res.get("ok"):
            raise ValueError(
                f"Failed to parse component '{p_strip}': {res.get('error')}"
            )
        parsed_parts.append(parse_preprocessed(res["result"]))
    if len(parsed_parts) != len(ops) + 1:
        raise ValueError("Invalid inequality structure.")
    op_map = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}
    relations = []
    for i, op_str in enumerate(ops):
        op_func = op_map.get(op_str)
        if not op_func:
            raise ValueError(f"Unknown operator: {op_str}")
        relations.append(op_func(parsed_parts[i], parsed_parts[i + 1]))
    return sp.And(*relations) if len(relations) > 1 else relations[0]


def solve_inequality(ineq_str: str, find_var: str | None = None) -> dict[str, Any]:
    """
    Solve an inequality.

    Args:
        ineq_str: Inequality string (e.g., "x > 0", "1 < x < 5")
        find_var: Optional variable to solve for

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: "inequality"
            - solutions: Dictionary mapping variable names to solution strings
            - error: Error message if ok is False
    """
    try:
        parsed = _parse_relational_fallback(ineq_str)
    except (ParseError, ValidationError) as e:
        logger.warning("Failed to parse inequality", exc_info=True)
        return {"ok": False, "error": f"Parse error: {e}", "error_code": "PARSE_ERROR"}
    except (ValueError, TypeError) as e:
        logger.warning("Type error parsing inequality", exc_info=True)
        return {
            "ok": False,
            "error": f"Invalid inequality: {e}",
            "error_code": "INVALID_INEQUALITY",
        }
    except Exception as e:
        logger.error("Unexpected error parsing inequality", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to parse inequality: {e}",
            "error_code": "PARSE_ERROR",
        }
    free_syms = list(parsed.free_symbols) if hasattr(parsed, "free_symbols") else []
    if find_var:
        target_sym = None
        for sym in free_syms:
            if str(sym) == find_var:
                target_sym = sym
                break
        if target_sym is None:
            return {
                "ok": False,
                "error": f"Variable '{find_var}' not found in the expression.",
            }
        vars_to_solve = [target_sym]
    else:
        if not free_syms:
            try:
                is_true = sp.simplify(parsed)
                return {
                    "ok": True,
                    "type": "inequality",
                    "solutions": {"result": str(is_true)},
                }
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(
                    f"Error finding variables in inequality: {e}", exc_info=True
                )
                return {
                    "ok": False,
                    "error": "No variable found in inequality",
                    "error_code": "NO_VARIABLE",
                }
        vars_to_solve = free_syms
    results = {}
    for v in vars_to_solve:
        try:
            ineqs_to_solve = (
                list(parsed.args) if isinstance(parsed, sp.And) else [parsed]
            )
            sol = sp.reduce_inequalities(ineqs_to_solve, v)
            results[str(v)] = str(sol)
        except NotImplementedError:
            logger.info(f"Inequality solving not implemented for {v}")
            results[str(v)] = "Solver not implemented for this type of inequality."
        except (ValueError, TypeError) as e:
            logger.warning(f"Error solving inequality for {v}", exc_info=True)
            results[str(v)] = f"Error solving for {v}: {e}"
        except Exception as e:
            logger.error(f"Unexpected error solving inequality for {v}", exc_info=True)
            results[str(v)] = f"Unexpected error solving for {v}: {e}"
    return {"ok": True, "type": "inequality", "solutions": results}
