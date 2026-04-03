"""
Handler functions for discovery commands (find, find ode, discover causal, etc.).
Part of the CLI refactoring to decompose repl_commands.py.
"""
import re
import logging
from typing import Any, Callable, Dict, Optional

import kalkulator_pkg.parser as kparser

logger = logging.getLogger(__name__)

# Regex patterns for data point and find-command parsing
FIND_PATTERN = re.compile(r"(?:find\s+)?([a-zA-Z_]\w*)(?:\s*\(([^)]*)\))?\s*$", re.IGNORECASE)
POINT_PATTERN = re.compile(r"([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)$")


def handle_find_ode(text: str):
    """Handle 'find ode' command for SINDy-based ODE discovery."""
    print("Note: 'find ode' requires data in specific format.")
    print("Usage: find ode from x=[...], dx_dt=[...]")
    print("This feature is experimental.")


def handle_discover_causal(text: str):
    """Handle 'discover causal' command for causal discovery."""
    print("Note: 'discover causal' is an experimental feature.")
    print("Usage: discover causal from <data>")


def handle_find_dimensionless(text: str):
    """Handle 'find dimensionless' command for dimensionless analysis."""
    print("Note: 'find dimensionless' is an experimental feature.")
    print("Usage: find dimensionless from <variables with units>")


def handle_find_command(text: str, variables: Dict[str, str]):
    """Handle standalone 'find f(x)' command."""
    content = text[5:].strip()  # Remove 'find '

    if "(" in content and ")" in content:
        match = re.match(r"([a-zA-Z_]\w*)\s*\(", content)
        if match:
            match.group(1)
            pass

    print("Function finding logic detected.")
    if "given" not in text and "=" not in text:
        print("Usage: f(1)=1, f(2)=4, find f(x)")
        print("       (Please provide data points in the same line)")


def handle_find_command_raw(
    text: str,
    ctx: Any,
    evolve_callback: Optional[Callable] = None,
) -> bool:
    """
    Handle 'find' command with integrated data points.
    e.g. "f(1)=2, f(2)=3, find f(x)"
    Returns True if handled.
    
    Args:
        evolve_callback: Optional callback to _handle_evolve for auto-fallback.
    """
    # 1. Split parts
    parts = kparser.split_top_level_commas(text)

    data_points = []
    target_func = None
    target_vars = []

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Strip flag for parsing
        p_clean = p.replace("--auto-evolve", "").strip()

        # Check for FIND command
        m_find = FIND_PATTERN.match(p_clean)
        if m_find and "find" in p_clean.lower():
            target_func = m_find.group(1)
            if m_find.group(2):
                target_vars = [v.strip() for v in m_find.group(2).split(",")]
            continue

        # Check for DATA point
        m_point = POINT_PATTERN.match(p_clean)
        if m_point:
            name = m_point.group(1)
            args_str = m_point.group(2)
            val_str = m_point.group(3)

            args = [a.strip() for a in args_str.split(",")]

            # PASTE ERROR DETECTION: Check if val_str contains another function call
            concat_match = re.search(r'^([0-9.\-+eE]+)([a-zA-Z_]\w*\s*\([^)]+\)\s*=.+)$', val_str)
            if concat_match:
                actual_val = concat_match.group(1)
                remaining = concat_match.group(2)
                
                data_points.append((name, args, actual_val))
                
                m_remaining = POINT_PATTERN.match(remaining)
                if m_remaining:
                    r_name = m_remaining.group(1)
                    r_args = [a.strip() for a in m_remaining.group(2).split(",")]
                    r_val = m_remaining.group(3)
                    data_points.append((r_name, r_args, r_val))
            else:
                data_points.append((name, args, val_str))

    if target_func and data_points:
        # Filter points for target function
        relevant_points = []
        for name, args, val in data_points:
            if name == target_func:
                relevant_points.append((args, val))

        if not relevant_points:
            print(f"No data points found for function '{target_func}'.")
            return True

        print(
            f"Finding function '{target_func}' from {len(relevant_points)} data points..."
        )

        # Infer vars if not provided
        if not target_vars:
            arity = len(relevant_points[0][0])
            defaults = ["x", "y", "z", "t", "u", "v"]
            target_vars = defaults[:arity]

        from ...function_manager import define_function
        from ...function_manager import find_function_from_data

        # Validate points structure
        valid_points = []
        for i, pt in enumerate(relevant_points):
            if not isinstance(pt, (tuple, list)):
                print(f"Skipping malformed point #{i}: {pt} (not tuple/list)")
                continue
            if len(pt) != 2:
                print(f"Skipping malformed point #{i}: {pt} (length {len(pt)})")
                continue
            valid_points.append(pt)
        relevant_points = valid_points

        # Handle unpacking safely
        try:
             result = find_function_from_data(ctx, relevant_points, target_vars)
        except ValueError as e:
             print(f"Regression Engine Crash: {e}. Data sample: {relevant_points[:3]}")
             return True
        if len(result) == 4:
            success, result_str, factored, error_msg = result
        elif len(result) == 3:
            success, result_str, error_msg = result
        else:
            success = False
            result_str = None
            error_msg = f"Internal API Error: Unexpected return length {len(result)}"

        if success:
            note = error_msg if error_msg else ""
            print(
                f"Discovered: {target_func}({', '.join(target_vars)}) = {result_str} {note}"
            )

            # Auto-fallback to Genetic Engine if confidence is low
            if "LOW CONFIDENCE" in str(note) and evolve_callback:
                print(
                    "Confidence too low. Switching to Genetic Engine (evolve) for robust discovery..."
                )

                points_str_list = []
                for args, val in relevant_points:
                    points_str_list.append(f"{target_func}({','.join(args)})={val}")
                data_str = ", ".join(points_str_list)

                evolve_cmd = (
                    f"evolve {target_func}({','.join(target_vars)}) from {data_str} --hybrid"
                )

                evolve_callback(evolve_cmd, ctx, variables=None)
                return True

            try:
                define_function(ctx, target_func, target_vars, result_str)
            except Exception as e:
                print(f"Warning: Failed to define function '{target_func}': {e}")
        else:
            auto_evolve = "--auto-evolve" in text.lower()

            if auto_evolve and evolve_callback:
                print(
                    f"Genius Mode failed ({error_msg}). Auto-switching to Evolve Mode..."
                )

                points_str_list = []
                for args_list, val_str in relevant_points:
                    args_joined = ",".join(args_list)
                    points_str_list.append(f"{target_func}({args_joined})={val_str}")

                points_segment = ", ".join(points_str_list)
                evolve_cmd = f"evolve {target_func}({','.join(target_vars)}) from {points_segment}"

                evolve_callback(evolve_cmd, ctx)
            else:
                print(f"Failed to discover function: {error_msg}")
                print(
                    f"Tip: Genius Mode seeks exact laws. Try 'evolve {target_func}({','.join(target_vars)})...' for approximate models."
                )
                print("     Or use '--auto-evolve' to switch automatically.")

        return True



    return False
