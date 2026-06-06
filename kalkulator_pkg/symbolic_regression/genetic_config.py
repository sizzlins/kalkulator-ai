"""Configuration for Genetic Symbolic Regression."""
from dataclasses import dataclass, field

@dataclass
class GeneticConfig:
    """Configuration for Genetic Symbolic Regression."""

    # Evolution Parameters
    population_size: int = 200
    n_islands: int = 2
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    parsimony_coefficient: float = 0.005
    max_depth: int = 8
    max_tree_depth: int = 15             # Hard cap on tree depth during evaluation
    
    early_stop_mse: float = 1e-10
    perfect_fit_threshold: float = 1e-9  # For "equivalent forms" check
    complexity_limit: int = 50           # Kill trees larger than this
    max_nested_powers: int = 5           # Prevent x**y**z**... hangs
    vertex_bonus: float = 5.0            # Weight boost for x=0
    anchor_bonus: float = 3.0            # Weight boost for integer anchors
    min_valid_ratio: float = 0.9         # Min % of valid data points
    min_valid_ratio: float = 0.9         # Min % of valid data points
    integer_tolerance: float = 1e-5      # Tolerance for integer anchor detection
    learning_rate: float = 1.0           # Boosting learning rate (1.0 = Full Model, 0.1 = Gradient Boosting)
    
    # Operators
    operators: list[str] = field(
        default_factory=lambda: [
            # Core arithmetic
            "add", "sub", "mul", "div", "pow",
            # Basic transcendental
            "sin", "cos", "tan", "exp", "log",
            "sqrt", "abs", "neg", "inv",
            # Hyperbolic
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            # Rounding/Step functions
            "floor", "ceil", "sign", "max", "min", "round", "frac", "trunc",
            # Inverse trig
            "atan", "asin", "acos",
            # Power shortcuts
            "square", "cube",
            # Protected operators (prevent NaN/complex)
            "plog", "psqrt",
            # Special functions
            "lambertw", "erf", "sinc", "heaviside",
            "bessel_j0", "bessel_j1", "gamma", "factorial",
            # Sequences
            "fibonacci", "lucas", "prime_pi", "ith_prime", "prime",
            # Bitwise (for integer patterns)
            "bitwise_xor", "bitwise_and", "bitwise_or", "lshift", "rshift",
            # Other binary
            "mod", "atan2",
        ]
    )

    
    # Weighted Complexity (Penalize "cheating" operators)
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {
            # Discontinuities (high penalty)
            "max": 2.0, "min": 2.0, "abs": 4.0, "heaviside": 4.0,
            # Steps/Rounding (Aggressively encouraged for discrete logic)
            "floor": 0.1, "ceil": 0.1, "round": 0.1, "frac": 0.1, "sign": 0.1, "trunc": 0.1,
            
            # Standard operations
            "pow": 1.0, "exp": 1.0, "log": 1.0, "plog": 1.0,
            "add": 1.0, "sub": 1.0, "mul": 1.0, "div": 1.0,
            "sqrt": 1.0, "psqrt": 1.0, "square": 1.0, "cube": 1.0,
            
            # Trig (standard)
            "sin": 1.0, "cos": 1.0, "tan": 1.0,
            "asin": 1.0, "acos": 1.0, "atan": 1.0, "atan2": 1.5,
            
            # Hyperbolic
            "sinh": 1.0, "cosh": 1.0, "tanh": 1.0,
            "asinh": 1.0, "acosh": 1.0, "atanh": 1.0,
            
            # Special functions (slightly higher - exotic)
            "lambertw": 2.0, "erf": 1.5, "sinc": 1.5,
            "bessel_j0": 2.0, "bessel_j1": 2.0, "gamma": 2.0, "factorial": 2.0,
            
            # Sequences (moderate penalty - discrete)
            "fibonacci": 1.0, "lucas": 1.0, "prime_pi": 1.0, "ith_prime": 1.0, "prime": 1.0,
            
            # Bitwise (encouraged for patterns)
            "bitwise_xor": 0.5, "bitwise_and": 0.5, "bitwise_or": 0.5,
            "lshift": 0.5, "rshift": 0.5,
            
            # Modulo
            "mod": 0.5,
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
    use_integer_anchoring: bool = False  # IntegerBiasWeighting: Bias towards integer points (formerly "Vise Strategy")
    use_integer_patterns: bool = False   # Bias towards integer constants (via LLL)
    elitism: int = 5
    boosting_rounds: int = 1
    high_precision: bool = False
    high_precision: bool = False
    n_jobs: int = 1  # 1=serial, >1=parallel workers, -1=all cores
    allow_bitwise: bool = True  # If False, disables bitwise operators (Continuity Shield)
    
    # Random Seed
    random_state: int | None = None
    
    # Loss Function (Robust Regression)
    loss_function: str = "mse" # "mse" or "huber"
    huber_delta: float = 1.35
