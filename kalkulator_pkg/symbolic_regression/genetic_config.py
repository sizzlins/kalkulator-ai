"""Configuration for Genetic Symbolic Regression."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class GeneticConfig:
    """Configuration for Genetic Symbolic Regression."""

    # Evolution Parameters
    population_size: int = 200
    n_islands: int = 2
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    parsimony_coefficient: float = 0.01
    max_depth: int = 8
    
    early_stop_mse: float = 1e-10
    perfect_fit_threshold: float = 1e-9  # For "equivalent forms" check
    complexity_limit: int = 100          # Kill trees larger than this
    max_nested_powers: int = 5           # Prevent x**y**z**... hangs
    vertex_bonus: float = 5.0            # Weight boost for x=0
    anchor_bonus: float = 3.0            # Weight boost for integer anchors
    min_valid_ratio: float = 0.9         # Min % of valid data points
    integer_tolerance: float = 1e-5      # Tolerance for integer anchor detection
    
    # Operators
    operators: list[str] = field(
        default_factory=lambda: [
            "add", "sub", "mul", "div", "pow",
            "sin", "cos", "tan", "exp", "log",
            "sqrt", "abs", "neg", "inv",
            # Advanced
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "floor", "ceil", "sign", "max", "min",
        ]
    )
    
    # Weighted Complexity (Penalize "cheating" operators)
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {
            "max": 5.0, "min": 5.0, "abs": 4.0,  # Discontinuities
            "floor": 2.0, "ceil": 2.0,           # Steps
            "pow": 1.0, "exp": 1.0, "log": 1.0,  # Standard
            "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 1.0
        }
    )
    default_complexity_weight: float = 1.0
    
    # Runtime
    timeout: float | None = 60.0
    verbose: bool = True
    seeds: list[str] = field(default_factory=list)
    
    # Strategies
    patience: int = 10
    min_improvement: float = 0.01
    constant_optimization_rate: float = 0.1
    migration_rate: float = 0.1
    migration_interval: int = 10
    
    # Heuristics (Opt-in to prevent bias)
    use_integer_anchoring: bool = False  # "Vise Strategy": Bias towards integer points
    use_integer_patterns: bool = False   # Bias towards integer constants (via LLL)
    elitism: int = 5
    boosting_rounds: int = 1
    high_precision: bool = False
    n_jobs: int = 1  # 1=serial, >1=parallel workers, -1=all cores
