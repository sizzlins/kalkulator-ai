![Demo](demo.png)

# Kalkulator-ai

![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.3.0-blue.svg)
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

# Optional: For Excel/Parquet support
pip install pandas openpyxl
```

## Usage

Legacy CLI:

```bash
python kalkulator.py
```

Or

Download the .exe file, kalkulator.exe
https://github.com/sizzlins/kalkulator-ai/blob/main/kalkulator.exe

### Commands

- `f(x)=...`: Define function.
- `find f(x)`: Discover function from data.
- `evolve ...`: Genetic programming search.
  - `--boost`: Enable 5-round boosting for complex models.
  - `--file <path>`: Load data from CSV/Excel/Parquet.
  - `--verbose`: Show progress.
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

### Advanced Data Input

**File Import (CSV/Excel)**

```bash
>>> evolve --file data.xlsx y = f(x)
Loaded 2 variables from 'data.xlsx': ['x', 'y']
```

**Explicit Target Syntax**

```bash
>>> evolve y = f(x) from x=[1,2,3], y=[3,5,7]
Discovered: 2*x + 1
```

**Boosting Mode (Hard Problems)**

```bash
>>> evolve --boost f(x) ...
Boost mode: 5x resources
```

### Calculus

**Differentiation**
Syntax: `diff(expression, variable)`

- **Why the variable?** In multi-variable calculus, you must specify which variable changes.
  - `diff(x*y, x)` -> `y` (Slope with respect to x)
  - `diff(x*y, y)` -> `x` (Slope with respect to y)

```
>>> diff(log(x), x)
Result: diff(log(x), x) = 1/x
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

### Numerical Limits & Precision

**Floating Point Limits (Machine Epsilon):**
Standard 64-bit floating point math breaks down when values differ by more than 15 decimal places ("Catastrophic Cancellation").

**Example:** `f(x) = (1+x)^(1/x)` (approaches `e` as x→0).

- `f(1e-10) ≈ 2.718` (Correct)
- `f(1e-16)` → `1.0` (Incorrect)

**Why**: `1.0 + 1e-16` is exactly `1.0` in computer memory. Then `1.0^Huge` is `1.0`. The tiny `x` information is lost before the exponentiation happens.

**High-Power Polynomials:**
Polynomials (`x^10`, `x^11`)...

- `find` works for `x^10` but may struggle with `x^11` and above (values exceed 10^11)
- `evolve` finds approximate exponents (e.g., `x^10.16` instead of `x^10`)

**Workaround:** Use `find` for high-power polynomials with moderate data ranges.

### Composite Functions

Deeply nested composite functions are beyond algorithm scope:

| Pattern               | Example        | Why                            |
| --------------------- | -------------- | ------------------------------ |
| Trig of rational      | `sin(1/(x-3))` | Infinite nesting possibilities |
| Nested transcendental | `sin(cos(x))`  | Combinatorial search space     |

**Workaround:** Define manually: `f(x)=sin(cos(tan(x)))`

### Complex Rational Functions

Multi-pole rational functions with polynomial numerators/denominators:

| Pattern         | Example         | Why                          |
| --------------- | --------------- | ---------------------------- |
| Quadratic ratio | `(1+x²)/(1-x²)` | Multiple poles, complex form |
| Higher-order    | `(x³+1)/(x³-1)` | No general templates         |

**Workaround:** Define manually: `f(x)=(1+x^2)/(1-x^2)`

### Square Root Functions

Nested radical functions involving squares:

| Pattern      | Example           | Why                     |
| ------------ | ----------------- | ----------------------- |
| Sqrt of quad | `sqrt(x²-16)`     | No sqrt(poly) templates |
| Nested sqrt  | `sqrt(x+sqrt(x))` | Infinite nesting        |

**Workaround:** Define manually: `f(x)=sqrt(x^2-16)`

### High-Degree Polynomials

Polynomials beyond degree 2:

| Pattern           | Example     | Why                                       |
| ----------------- | ----------- | ----------------------------------------- |
| Degree 3+         | `3x⁵ - 5x³` | No x³/x⁴/x⁵ templates; too complex for GP |
| High coefficients | `100x³`     | Large search space                        |

**Workaround:** Define manually: `f(x)=3x^5-5x^3`

## Advanced: External Tools

For complex functions our algorithms can't discover (e.g., `sqrt(x²-16)`), consider using **PySR** externally:

```bash
pip install pysr  # Requires Julia (~500MB download)
```

```python
from pysr import PySRRegressor
import numpy as np

X = np.array([[4],[5],[6],[7],[8]])
y = np.array([0, 3, 4.47, 5.74, 6.93])

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "sin", "cos"],
)
model.fit(X, y)
print(model)  # Shows discovered equations
```

> **Note:** PySR is a separate project. First run downloads Julia and takes 1-2 minutes.

## Floating-Point Precision Limitations

⚠️ **Important:** Kalkulator uses standard IEEE 754 double-precision floating-point arithmetic, which has **~15-17 decimal digits of precision**.

### What This Means

Values beyond this precision are **automatically rounded**:

```
>>> 1.000000000000000000000000000000000000000001
Result: 1

>>> 1 = 0.9999999999999999999999999999999999999999999
Result: Identity  # Both round to exactly 1.0
```

This is **not a bug** - it's a fundamental limitation of how computers represent real numbers. The extra digits beyond position 15-17 are lost during parsing.

### When This Matters

- **Cryptography**: Use arbitrary-precision libraries (`decimal.Decimal`, SymPy `Rational`)
- **Financial calculations**: Consider using integer cents instead of fractional dollars
- **Exact symbolic math**: Define expressions symbolically (e.g., `1/3` not `0.333333...`)

### When It Doesn't Matter

- **Scientific calculations**: 15 digits is sufficient for most physics/engineering
- **Data analysis**: Machine learning models don't need beyond double precision
- **Everyday math**: Calculator-style operations

For true arbitrary precision, consider using SymPy's `Rational` or Python's `decimal.Decimal` module directly.

## Architecture

- **Core**: `kalkulator_pkg`
- **Solvers**: Hybrid Sparse Regression (Lasso/OMP) + Genetic Programming + SINDy.
- **Safety**: Bounded execution, sandboxed evaluation.

## License

MIT.
