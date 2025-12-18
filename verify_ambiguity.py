
from kalkulator_pkg.cli.repl_commands import _handle_evolve
from kalkulator_pkg.cli.repl_core import REPL
import numpy as np

def test_ambiguity():
    print("Testing evolve variable ambiguity...")
    
    # Setup context:
    # x = [1, 2, 3]
    # y = [2, 4, 6]  (Target: 2x)
    # z = [10, 20, 30] (Target: 10x)
    
    # If we run 'evolve f(x)', which one does it pick?
    
    variables = {
        'x': np.array([1, 2, 3]),
        'z': np.array([10, 20, 30]),
        'y': np.array([2, 4, 6])
    }
    
    # Force 'z' to be first in iteration if dict is insertion ordered? 
    # Python 3.7+ preserves insertion order. 
    # If z was added before y, data_dict keys will be ['x', 'z', 'y'].
    # output_candidates will be ['z', 'y'].
    # It will pick 'z'.
    
    print("Variables keys order:", list(variables.keys()))
    
    # Mocking print to capture output
    output = []
    original_print = __builtins__['print']
    
    def mock_print(*args, **kwargs):
        msg = " ".join(map(str, args))
        output.append(msg)
        original_print("CAPTURED:", msg)
        
    __builtins__['print'] = mock_print
    
    try:
        # We need to monkeypatch the internal print of repl_commands if it uses the builtin one
        # It's in the same process, so overriding builtin print should work.
        
        _handle_evolve("evolve f(x)", variables)
        
    finally:
        __builtins__['print'] = original_print
        
    # check results
    fit_2x = False
    fit_10x = False
    
    for line in output:
        if "Result:" in line:
            if "2*" in line.replace(" ", ""): # 2*x
                 fit_2x = True
            elif "10*" in line.replace(" ", ""): # 10*x
                 fit_10x = True
                 
    if fit_10x and not fit_2x:
        print("FAIL: Evolve picked 'z' (10x) instead of 'y' (2x) just because of order!")
    elif fit_2x:
        print("SUCCESS: Evolve picked 'y' correctly.")
    else:
        print("INCONCLUSIVE: Found neither?")

if __name__ == "__main__":
    test_ambiguity()
