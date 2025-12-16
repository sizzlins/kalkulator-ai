![Demo](demo.png)

# Kalkulator-ai

![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

Symbolic regression engine. Discovers mathematical formulas, ODEs, and causal relationships from data.

## Capabilities

- **Symbolic Regression**: Identifies exact equations (`y = 5*x^2`) from raw data.
- **Physics Discovery**: Patterns include inverse square laws, exp/log, and trigonometric series.
- **SINDy**: Discovers differential equations (`dx/dt`) from time-series.
- **Genetic Programming**: Evolves complex functional forms via mutation/crossover.
- **Calculus**: Symbolic differentiation (`diff`) and integration (`integrate`).
- **Agentic Discovery**: Intelligent feature selection logic.

## Installation

Requires Python 3.8+.

```bash
git clone https://github.com/sizzlins/kalkulator-ai
pip install -r requirements.txt
```

## Usage

Legacy CLI:

```bash
python kalkulator.py
```

### Commands

- `f(x)=...`: Define function.
- `find f(x)`: Discover function from data.
- `evolve ...`: Genetic programming search.
- `diff(...)`: Differentiate.
- `integrate(...)`: Integrate.
- `save/load`: Persist state.

### Examples

**Physics (Kinetic Energy)**

```
>>> E(2,4)=16, E(4,2)=8, E(10,1)=5, find E(m,v)
E(m, v) = 0.5*m*v^2
```

**LambertW (Inverse x^x)**

```
>>> f(4)=2, f(27)=3, f(256)=4, find f(x)
Discovered: f(x) = exp(LambertW(log(x)))
```

**Gaussian**

```
>>> g(0)=1, g(1)=0.3679, find g(x)
g(x) = exp(-x^2)
```

## Limitations

### Function Types

Function finding discovers **continuous mathematical relationships**. The following are **not auto-discoverable**:

| Type                   | Examples                        | Reason                       |
| ---------------------- | ------------------------------- | ---------------------------- |
| Discrete/Combinatorial | `factorial(x)`, `binomial(n,k)` | Integer-only, not in physics |
| Piecewise              | `abs(x)`, step functions        | Discontinuous                |
| Recursive              | Fibonacci, Ackermann            | No closed-form               |

**Workaround:** Define manually: `f(x)=x!`

### When to Use `find` vs `evolve`

| Use Case                    | Recommended | Why                              |
| --------------------------- | ----------- | -------------------------------- |
| Clean data, known patterns  | `find`      | Exact regression is reliable     |
| Exponential (`2^x`, `e^x`)  | `find`      | Has explicit `exp(a*x)` template |
| Pole functions (`x/(x-1)²`) | `find`      | Auto-detects poles from inf/nan  |
| Noisy/complex data          | `evolve`    | Exploratory, tolerates noise     |
| Novel function forms        | `evolve`    | Searches without assumptions     |

**Tips:**

- Use `find` first for most cases - it's faster and more reliable
- `evolve` auto-seeds with detected patterns (poles, frequencies) for better results
- If `find` gives low confidence, try `evolve` for exploration

### Data Types

Regression (`find`, `evolve`) requires **real-valued** inputs and outputs. Complex data is automatically filtered with a warning:

```
>>> f(-4),f(-3),...  # Some values are complex: -5.5 - 12.5i
>>> f(-4)=..., find f(x)
Warning: 4 data point(s) with complex/imaginary values were skipped.
         Regression requires real-valued inputs and outputs.
```

**Workaround:** Use only real-valued data points for regression.

### Numerical Limits

High-power polynomials (`x^10`, `x^11`, etc.) may hit numerical precision limits:

- `find` works for `x^10` but may struggle with `x^11` and above (values exceed 10^11)
- `evolve` finds approximate exponents (e.g., `x^10.16` instead of `x^10`)

**Workaround:** Use `find` for high-power polynomials with moderate data ranges.

### Composite Functions

Deeply nested composite functions are beyond algorithm scope:

| Pattern               | Example        | Why                            |
| --------------------- | -------------- | ------------------------------ |
| Trig of rational      | `sin(1/(x-3))` | Infinite nesting possibilities |
| Nested transcendental | `exp(sin(x))`  | Combinatorial search space     |

**Workaround:** Define manually: `f(x)=sin(4/(x-3))`

### Complex Rational Functions

Multi-pole rational functions with polynomial numerators/denominators:

| Pattern         | Example         | Why                          |
| --------------- | --------------- | ---------------------------- |
| Quadratic ratio | `(1+x²)/(1-x²)` | Multiple poles, complex form |
| Higher-order    | `(x³+1)/(x³-1)` | No general templates         |

**Workaround:** Define manually: `f(x)=(1+x^2)/(1-x^2)`

## Architecture

- **Core**: `kalkulator_pkg`
- **Solvers**: Hybrid Sparse Regression (Lasso/OMP) + Genetic Programming + SINDy.
- **Safety**: Bounded execution, sandboxed evaluation.

## License

MIT.
