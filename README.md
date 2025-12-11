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

## Architecture

- **Core**: `kalkulator_pkg`
- **Solvers**: Hybrid Sparse Regression (Lasso/OMP) + Genetic Programming + SINDy.
- **Safety**: Bounded execution, sandboxed evaluation.

## License

MIT.
