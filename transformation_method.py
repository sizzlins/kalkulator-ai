"""
Add this at the end of genetic_engine.py (after line 1113):

This is the variable transformation search method that runs evolution
in multiple mathematical spaces simultaneously.
"""

def fit_with_transformations(
    self,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str] | None = None,
) -> tuple[str, float, str]:
    """
    Run evolution in multiple transformed spaces simultaneously.
    
    This makes complex functions discoverable by transforming them into
    simpler spaces. For example, (1+x)^(1/x) is complex in direct space
    but becomes (1/x)*log(1+x) in log-space (simple multiplication!).
    
    Transformations attempted:
    1. Direct: y = f(x)
    2. Log: log(y) = g(x), then y = exp(g(x))
    3. Inverse: 1/y = h(x), then y = 1/h(x)
    
    Args:
        X: Input data
        y: Target values
        variable_names: Names for variables
        
    Returns:
        Tuple of (best_expression, best_mse, best_space_name)
    """
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(X.shape[1])]
    
    spaces = []
    
    # Space 1: Direct (always valid)
    spaces.append({
        'name': 'direct',
        'X': X,
        'y': y,
        'transform_back': lambda expr: expr,
        'filter': np.ones(len(y), dtype=bool)
    })
    
    # Space 2: Log (only if y > 0)
    y_positive_mask = y > 1e-10
    if np.sum(y_positive_mask) > len(y) * 0.5:  # Need majority positive
        spaces.append({
            'name': 'log',
            'X': X[y_positive_mask],
            'y': np.log(np.maximum(y[y_positive_mask], 1e-10)),
            'transform_back': lambda expr: f"exp({expr})",
            'filter': y_positive_mask
        })
    
    # Space 3: Inverse (only if y != 0)
    y_nonzero_mask = np.abs(y) > 1e-10
    if np.sum(y_nonzero_mask) > len(y) * 0.5:
        spaces.append({
            'name': 'inverse',
            'X': X[y_nonzero_mask],
            'y': 1.0 / y[y_nonzero_mask],
            'transform_back': lambda expr: f"1/({expr})",
            'filter': y_nonzero_mask
        })
    
    if self.config.verbose:
        print(f"Running evolution in {len(spaces)} spaces: {[s['name'] for s in spaces]}")
    
    # Run evolution in each space
    results = []
    for space in spaces:
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"Evolving in {space['name'].upper()} space...")
            print(f"{'='*70}")
        
        # Create fresh regressor for this space
        temp_regressor = GeneticSymbolicRegressor(self.config)
        temp_regressor.fit(space['X'], space['y'], variable_names)
        
        # Get best expression
        best_expr = temp_regressor.get_best_expression()
        
        # Transform back to original space
        if space['name'] != 'direct':
            transformed_expr = space['transform_back'](best_expr)
        else:
            transformed_expr = best_expr
        
        # Evaluate MSE in ORIGINAL space (ALL data points)
        try:
            # Parse and evaluate the transformed expression
            from ..parser import parse_preprocessed
            import sympy as sp
            
            # Create symbol dict
            symbols = {name: sp.Symbol(name) for name in variable_names}
            parsed = sp.sympify(transformed_expr, locals=symbols)
            
            # Evaluate on all original data
            pred = []
            for i in range(len(X)):
                var_dict = {variable_names[j]: X[i, j] for j in range(X.shape[1])}
                try:
                    val = float(parsed.subs(var_dict))
                    pred.append(val)
                except:
                    pred.append(np.nan)
            
            pred = np.array(pred)
            valid_mask = ~np.isnan(pred)
            if np.sum(valid_mask) > 0:
                mse = np.mean((pred[valid_mask] - y[valid_mask])**2)
            else:
                mse = 1e10
        except Exception:
            mse = 1e10
        
        results.append({
            'space': space['name'],
            'expression': transformed_expr,
            'mse': mse,
            'original_expr': best_expr
        })
        
        if self.config.verbose:
            print(f"\nResult in {space['name']} space:")
            print(f"  Original: {best_expr}")
            if space['name'] != 'direct':
                print(f"  Transformed: {transformed_expr}")
            print(f"  MSE in original space: {mse:.6e}")
    
    # Return best across all spaces
    best = min(results, key=lambda r: r['mse'])
    
    if self.config.verbose:
        print(f"\n{'='*70}")
        print(f"BEST SPACE: {best['space'].upper()}")
        print(f"Expression: {best['expression']}")
        print(f"MSE: {best['mse']:.6e}")
        print(f"{'='*70}")
    
    return best['expression'], best['mse'], best['space']
