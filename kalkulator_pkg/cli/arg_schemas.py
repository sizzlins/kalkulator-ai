"""
Argument schemas and parsing for REPL commands.

Replaces the fragile regex-based flag extraction in _handle_evolve with
structured, type-safe argument parsing using shlex tokenization.

Usage:
    from kalkulator_pkg.cli.arg_schemas import parse_evolve_flags, EvolveConfig

    config, remaining_text = parse_evolve_flags(raw_text)
    # config.boost  -> int (default 1)
    # config.verbose -> bool
    # config.seeds  -> list[str]
    # remaining_text -> data portion only, no flags
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvolveConfig:
    """Typed configuration for the evolve command."""

    # Modes
    hybrid: bool = False
    verbose: bool = False
    super_verbose: bool = False
    debug: bool = False
    transform: bool = False
    high_precision: bool = False
    polynomial: bool = False
    discover_ode: bool = False

    # Parameterized flags
    boost: int = 1
    seeds: List[str] = field(default_factory=list)
    banned: List[str] = field(default_factory=list)
    file_path: Optional[str] = None


def parse_evolve_flags(text: str) -> tuple[EvolveConfig, str]:
    """Parse flags from evolve command text, returning config and remaining data.

    Uses shlex.split for proper tokenization (handles quoted strings).
    Separates --flag tokens from data tokens so the data portion is never
    corrupted by flag removal.

    Args:
        text: Raw command text (after 'evolve' keyword and shortcut expansion).

    Returns:
        Tuple of (EvolveConfig, remaining_text) where remaining_text has all
        flags removed and only contains function/data specification.

    Example:
        >>> config, data = parse_evolve_flags(
        ...     "f(x) from f(1)=2, f(2)=4 --boost 3 --verbose --seed 'x**2'"
        ... )
        >>> config.boost
        3
        >>> config.verbose
        True
        >>> config.seeds
        ['x**2']
        >>> data
        "f(x) from f(1)=2, f(2)=4"
    """
    config = EvolveConfig()

    # Special case: empty or whitespace-only input
    if not text or not text.strip():
        return config, text

    # --- Phase 1: Extract seed expressions (quoted, need special handling) ---
    # Seeds use: --seed 'expr' or --seed "expr"
    # We must extract these BEFORE shlex to avoid issues with commas in seeds
    import re
    seed_pattern = re.compile(r"""--seed\s+(['"])(.*?)\1""")
    seed_matches = seed_pattern.findall(text)
    for _, expr in seed_matches:
        config.seeds.append(expr)
    text = seed_pattern.sub("", text)

    # --- Phase 2: Tokenize remaining text ---
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        # Unmatched quotes — fall back to simple split
        tokens = text.split()

    data_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()

        # Boolean flags
        if token_lower == "--hybrid":
            config.hybrid = True
        elif token_lower == "--verbose":
            config.verbose = True
        elif token_lower in ("--super-verbose", "-sv"):
            config.super_verbose = True
        elif token_lower == "--debug":
            config.debug = True
        elif token_lower == "--transform":
            config.transform = True
        elif token_lower in ("--high-precision", "--hp"):
            config.high_precision = True
        elif token_lower == "--polynomial":
            config.polynomial = True
        elif token_lower == "--discover-ode":
            config.discover_ode = True

        # Parameterized flags (consume next token as value)
        elif token_lower == "--boost":
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                i += 1
                config.boost = int(tokens[i])
            else:
                config.boost = 5  # Default when flag present without value
        elif token_lower.startswith("--boost="):
            try:
                config.boost = int(token_lower.split("=", 1)[1])
            except (ValueError, IndexError):
                config.boost = 5

        elif token_lower == "--ban":
            if i + 1 < len(tokens):
                i += 1
                config.banned.extend(
                    f.strip().lower() for f in tokens[i].split(",") if f.strip()
                )

        elif token_lower == "--file":
            if i + 1 < len(tokens):
                i += 1
                config.file_path = tokens[i]

        elif token_lower == "--seed":
            # Seed without matching quote pair (already handled above).
            # Try to grab next token as seed expression.
            if i + 1 < len(tokens):
                i += 1
                config.seeds.append(tokens[i])

        else:
            # Not a flag — keep as data token
            data_tokens.append(token)

        i += 1

    # --- Phase 3: Reconstruct remaining text ---
    # Join data tokens back. We preserve the original spacing for the data
    # portion because the existing regex patterns expect specific formats
    # like "f(x) from f(1)=2, f(2)=4".
    #
    # Strategy: Walk the original text and remove only flag segments,
    # preserving everything else including commas, parens, and spacing.
    remaining = text
    # Remove all recognized flags from the original text string
    # This is more reliable than reconstructing from tokens because
    # shlex may have split data like "f(1)=2," differently.
    flag_patterns = [
        r"--hybrid\b",
        r"--verbose\b",
        r"--super-verbose\b",
        r"-sv\b",
        r"--debug\b",
        r"--transform\b",
        r"--high-precision\b",
        r"--hp\b",
        r"--polynomial\b",
        r"--discover-ode\b",
        r"--boost(?:\s+\d+|=\d+)?",
        r"--ban\s+[a-zA-Z0-9_,]+",
        r"""--file\s+['"]?[^'"\s]+['"]?""",
    ]
    for pat in flag_patterns:
        remaining = re.sub(pat, "", remaining, flags=re.IGNORECASE)

    # Clean up extra whitespace from removals
    remaining = re.sub(r"\s{2,}", " ", remaining).strip()

    return config, remaining
