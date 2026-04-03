
import sympy as sp

def _try_decompose_symbol(symbol_str: str, candidates: list[str]) -> list[str] | None:
    """Try to decompose a symbol string into a list of candidate strings (implicit multiplication).
    
    Greedy recursive matching.
    """
    if not symbol_str:
        return []
    
    # Sort candidates by length descending
    sorted_candidates = sorted(candidates, key=len, reverse=True)
    
    for cand in sorted_candidates:
        if symbol_str.startswith(cand):
            remainder = symbol_str[len(cand):]
            if not remainder:
                return [cand]
            
            suffix_parts = _try_decompose_symbol(remainder, candidates)
            if suffix_parts is not None:
                return [cand] + suffix_parts
                
    return None

def test_xy_parsing():
    print("Testing 'xy' parsing with Implicit Multiplication Logic...")
    
    # Setup
    x, y = sp.symbols('x y')
    # Initial state: xy is parsed as a single symbol because space was missing
    xy = sp.Symbol('xy')
    body = xy
    params = ['x', 'y']
    
    print(f"Initial Parse: {body}")
    
    # Simulation of define_function logic
    print("\nSimulating Auto-Fix Logic:")
    
    if hasattr(body, "free_symbols"):
        # Create a list because we might modify body, which changes free_symbols
        # But here body is just 'xy'
        # In real code, we iterate over a snapshot or handle substitution carefully
        symbols = list(body.free_symbols)
        for symbol in symbols:
            symbol_str = str(symbol)
            if symbol_str not in params:
                print(f"  Found unbound symbol: {symbol_str}")
                
                 # Check if it matches any param insensitively
                match_found = False
                for param in params:
                    if symbol_str.lower() == param.lower():
                        # Mismatch found (e.g. X vs x). Fix it.
                        print(f"    Replacing {symbol} with {param} (Match found)")
                        body = body.subs(symbol, sp.Symbol(param))
                        match_found = True
                        break
                
                # Logic for implicit multiplication (e.g. xy -> x*y)
                if not match_found:
                    print(f"    Attempting decomposition for {symbol}...")
                    parts = _try_decompose_symbol(symbol_str, params)
                    if parts:
                        print(f"    Decomposed into: {parts}")
                        # Replace symbol with Mul(*parts)
                        # Use sp.Symbol to ensure we use valid symbols
                        new_expr = sp.Mul(*[sp.Symbol(p) for p in parts])
                        body = body.subs(symbol, new_expr)
    
    print(f"Final Body: {body}")
    
    # Verify result
    try:
        # f(1, 2) should be 1*2 = 2
        result = body.subs({x: 1, y: 2})
        print(f"RESULT f(1,2): {result}")
    except Exception as e:
         print(f"RESULT ERROR: {e}")

if __name__ == "__main__":
    test_xy_parsing()
