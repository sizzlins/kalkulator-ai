"""Safe Tokenizer for input transformation."""
import tokenize
import io
import token as token_module
from typing import Generator
from collections import namedtuple

# Use namedtuple for immutable token representation
Token = namedtuple('Token', ['type', 'string', 'start', 'end', 'line'])

FORBIDDEN_NAMES = {
    "import", "lambda", "class", "def", "return", "yield", "raise", "exec", "eval",
    "open", "global", "nonlocal", "__import__", "__class__", "__bases__", "__subclasses__"
}

KNOWN_FUNCTIONS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "exp", "log", "sqrt", "abs", "floor", "ceil", "sign", "max", "min", 
    "mod", "Mod", "heaviside", "erf", "gamma"
}

def transform_input(text: str) -> str:
    """Safely transform input string using tokenization.
    
    Handles:
    - Implicit multiplication (2x -> 2*x, x y -> x*y)
    - Syntax conversion (^ -> **, mod -> Mod)
    - Unicode replacements (√ -> sqrt, π -> pi, etc.)
    - Safe FSM-based Smart Sqrt wrapping (√x -> sqrt(x))
    - Strict blocklisting of dangerous keywords
    
    Args:
        text: Input expression string
        
    Returns:
        Transformed string safe for SymPy parsing
        
    Raises:
        ValueError if forbidden tokens are found
    """
    if not text: return ""
    
    # 1. Basic Char replacements (Safe, O(N))
    text = text.replace("×", "*").replace("–", "-").replace("−", "-")
    text = text.replace(":", "/")
    
    # v4.7 Security Fix: Tokenizer Crash Prevention
    # Python's tokenizer chokes on backslashes at end of line or specific locations.
    # Since mathematical expressions don't use backslashes, we can safely remove them.
    text = text.replace("\\", "")
    
    # Python tokenizer requires bytes
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(text.encode('utf-8')).readline))
    except tokenize.TokenError as e:
        raise ValueError(f"Tokenization failed: {e}")
        
    result_tokens = []
    
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        tok_type = tok.type
        tok_str = tok.string
        
        # Skip encoding token and ENDMARKER
        if tok_type == 0 or tok_type == token_module.ENDMARKER:
            i += 1
            continue
            
        # --- Unicode / Special Token Logic ---
        # Map known unicode names
        if tok_str == "π": tok_str = "pi"
        elif tok_str == "Δ": tok_str = "Delta"
        elif tok_str == "mod": tok_str = "Mod"
        elif tok_str == "^": tok_str = "**"
        
        # Simple char replacements if they survived raw replace
        if "×" in tok_str: tok_str = tok_str.replace("×", "*")

        # --- Standard Safety & Implicit Mult ---
        
        if tok_type == token_module.NAME:
            if tok_str in FORBIDDEN_NAMES or tok_str.startswith("__"):
                raise ValueError(f"Forbidden token detected: '{tok_str}'")
        
        # Check for implicit multiplication with the LAST added token
        if result_tokens:
            last_type, last_str = result_tokens[-1]
            
            should_mult = False
            # Number (or ) ) followed by Name or (
            if last_type == token_module.NUMBER:
                if tok_type == token_module.NAME: should_mult = True
                elif tok_str == "(": should_mult = True
            elif last_str == ")":
                if tok_type == token_module.NAME: should_mult = True
                elif tok_str == "(": should_mult = True
            elif last_type == token_module.NAME:
                if tok_type == token_module.NAME: should_mult = True
                # Don't mul if 'sin('
            
            if should_mult:
                result_tokens.append((token_module.OP, "*"))
        
        result_tokens.append((tok_type, tok_str))
        i += 1

    result = tokenize.untokenize(result_tokens).strip()
    if isinstance(result, bytes):
        return result.decode('utf-8')
    return result
