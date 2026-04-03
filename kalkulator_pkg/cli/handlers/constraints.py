"""
Handler functions for constraint commands (ban/unban).
Part of the CLI refactoring to decompose repl_commands.py.
"""
from typing import Any


def matches_ban(func_lower: str, ban_token: str) -> bool:
    """Check if a function string violates a ban token.
    
    Handles:
    - Stripping parens from ban tokens: 'sqrt()' -> checks for 'sqrt'
    - Semantic equivalences: sqrt <-> x**0.5, pow <-> ** <-> ^
    - Unicode symbols: √ -> sqrt
    """
    # Normalize inputs
    func_lower = func_lower.lower()
    ban_token = ban_token.lower().replace("()", "")
    
    # Direct substring check
    if ban_token in func_lower:
        return True
    
    # Semantic equivalences
    SQRT_FORMS = {'sqrt', '**0.5', '**(0.5)', '**(1/2)', '**0.50'}
    POW_FORMS = {'pow', '**', '^'}
    
    if ban_token in {'sqrt', 'x**0.5'}:
        return any(form in func_lower for form in SQRT_FORMS)
    if ban_token in POW_FORMS:
        return any(form in func_lower for form in POW_FORMS)
    
    return False


def handle_ban_command(text: str, ctx: Any):
    """
    Handle 'ban' command to persistently exclude operators from evolution.
    Syntax: ban <func1>, <func2>, ...
    """
    if not ctx:
        print("Error: Context not available for persistent banning.")
        return

    # Extract arguments (remove 'ban ' prefix)
    args = text[3:].strip()
    if not args:
        # If no args, just list currently banned
        if ctx.banned_operators:
             print(f"Currently banned operators: {sorted(list(ctx.banned_operators))}")
        else:
             print("No operators are currently banned.")
        return

    # Parse comma-separated list
    new_bans = [op.strip().lower() for op in args.split(",") if op.strip()]
    
    ctx.banned_operators.update(new_bans)
    print(f"Banned: {sorted(list(ctx.banned_operators))}")


def handle_unban_command(text: str, ctx: Any):
    """
    Handle 'unban' command to remove exclusions.
    Syntax: unban <func> | unban all
    """
    if not ctx:
        print("Error: Context not available.")
        return

    # Normalize command to handle both "unban" and "unban <args>"
    # If just "unban", text is "unban" -> args empty -> error
    if text.strip().lower() == "unban":
        print("Usage: unban <function_name> | unban all")
        return

    args = text[5:].strip()
    if not args:
        print("Usage: unban <function_name> | unban all")
        return

    if args.lower() == "all":
        ctx.banned_operators.clear()
        print("All bans cleared.")
        return

    to_remove = [op.strip().lower() for op in args.split(",") if op.strip()]
    missing = []
    removed = []
    
    for op in to_remove:
        if op in ctx.banned_operators:
            ctx.banned_operators.remove(op)
            removed.append(op)
        else:
            missing.append(op)
            
    if removed:
        print(f"Unbanned: {removed}")
    if missing:
        print(f"Not found in ban list: {missing}")
