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
        """Initialize population using standard ramped half-and-half."""
        # We need to replicate the logic from EvolutionStrategy.initialize_population
        # Or better yet, we can't easily access Strategy here without cycle.
        # We will implement a basic initialization here since strict Strategy dependency is hard?
        # Actually, let's just use the logic directly.
        
        depths = range(2, self.config.max_depth + 1)
        methods = ["grow", "full"]
        
        self.clear()
        
        # 1. Seeds (skip for now or handle basic)
        
        # 2. Random Trees
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
