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
    - strict blocklisting of dangerous keywords
    
    Args:
        text: Input expression string
        
    Returns:
        Transformed string safe for SymPy parsing
        
    Raises:
        ValueError if forbidden tokens are found
    """
    if not text: return ""
    
    # Python tokenizer requires bytes
    tokens = list(tokenize.tokenize(io.BytesIO(text.encode('utf-8')).readline))
    result_tokens = []
    
    prev_token = None
    
    for tok in tokens:
        tok_type = tok.type
        tok_str = tok.string
        
        # Skip encoding token and ENDMARKER
        if tok_type == 0 or tok_type == token_module.ENDMARKER: # 0 is ENCODING
            continue
            
        # 1. Safety Check
        if tok_type == token_module.NAME:
            if tok_str in FORBIDDEN_NAMES or tok_str.startswith("__"):
                raise ValueError(f"Forbidden token detected: '{tok_str}'")
            
            # Syntax Sugar: mod -> Mod
            if tok_str == "mod":
                tok_str = "Mod"
                
        # 2. Operator Transformation
        if tok_type == token_module.OP:
            if tok_str == "^":
                tok_str = "**"
                
        # 3. Implicit Multiplication Logic
        if prev_token:
            should_insert_mult = False
            prev_type = prev_token.type
            prev_str = prev_token.string
            
            # Case A: Number followed by Name (2x) or Paren (2(x))
            if prev_type == token_module.NUMBER:
                if tok_type == token_module.NAME:
                    should_insert_mult = True
                elif tok_type == token_module.OP and tok_str == "(":
                    should_insert_mult = True
                    
            # Case B: Name followed by Name (x y) or Paren (x(y) -> Function or Mult?)
            elif prev_type == token_module.NAME:
                if tok_type == token_module.NAME:
                    # 'sin x' -> invalid in math usually, but imply mult 'sin*x'
                    # 'x y' -> 'x*y'
                    should_insert_mult = True
                elif tok_type == token_module.OP and tok_str == "(":
                    # 'sin(x)' -> Function Call (No mult)
                    # 'x(y)' -> Function Call (unknown func) or x*(y)?
                    # Standard math convention: x(y) is x*y if x is var, or func if x is func.
                    # We assume Function Call if it looks like one.
                    # BUT, SymPy parses 'x(y)' as Function('x')(y).
                    # '2x(y)' -> 2*x*y? Or 2*Function(x)(y)?
                    # For safety/simplicity, we assume Function Call.
                    should_insert_mult = False 
                    
            # Case C: Close Paren followed by Name or Paren: (a)b or (a)(b)
            elif prev_type == token_module.OP and prev_str == ")":
                 if tok_type == token_module.NAME:
                     should_insert_mult = True
                 elif tok_type == token_module.OP and tok_str == "(":
                     should_insert_mult = True
            
            if should_insert_mult:
                # Insert '*' op
                # Position info doesn't matter for untokenize
                result_tokens.append((token_module.OP, "*"))

        # Add current token
        result_tokens.append((tok_type, tok_str))
        prev_token = tok

    # Reconstruct string
    result = tokenize.untokenize(result_tokens)
    if isinstance(result, bytes):
        return result.decode('utf-8')
    return result
