"""Population class for Genetic Symbolic Regression."""
from .expression_tree import ExpressionTree
from .genetic_config import GeneticConfig
import random

class Population(list):
    """A population of ExpressionTrees (Island).
    
    Inherits from list to be compatible with legacy code expecting list[ExpressionTree].
    """
    def __init__(self, size: int, variable_names: list[str], config: GeneticConfig, random_state: int = None):
        super().__init__()
        self.size = size
        self.variable_names = variable_names
        self.config = config
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            
    def initialize(self, seeds: list[str] = None):
        """Initialize population with ramped half-and-half and optional seeds."""
        self.clear()
        
        # 1. Seed Injection
        if seeds:
            from ..parser import safe_sympy_parse
            from ..config import ALLOWED_SYMPY_NAMES
            import sympy as sp
            
            # Prepare local dict for parsing
            local_dict = {v: sp.Symbol(v) for v in self.variable_names}
            full_local_dict = {**ALLOWED_SYMPY_NAMES, **local_dict}
            
            injected_count = 0
            # Cap injection at 50% of population
            max_injected = max(1, self.size // 2)
            
            for seed_str in seeds:
                if injected_count >= max_injected: break
                try:
                    # v4.4: Use safe parser instead of unsafe sympify
                    # This standardizes parsing and prevents execution of malicious/toxic seeds
                    expr = safe_sympy_parse(seed_str, local_dict=full_local_dict)
                    # Convert to tree
                    tree = ExpressionTree.from_sympy(expr, self.variable_names)
                    if tree:
                        tree.age = 0
                        self.append(tree)
                        injected_count += 1
                except Exception:
                    # Ignore invalid seeds silently
                    pass
            
            if injected_count > 0:
                # We can't easily log from here without passing a logger or checking config.verbose
                # but that's okay.
                pass

        # 2. Random Trees (Ramped Half-and-Half)
        # Fill the rest of the population
        depths = range(2, self.config.max_depth + 1)
        methods = ["grow", "full"]
        
        while len(self) < self.size:
            depth = depths[len(self) % len(depths)]
            method = methods[len(self) % len(methods)]
            
            tree = ExpressionTree.random_tree(
                variables=self.variable_names,
                max_depth=depth,
                operators=self.config.operators if hasattr(self.config, 'operators') else None,
                method=method
            )
            tree.age = 0
            self.append(tree)
