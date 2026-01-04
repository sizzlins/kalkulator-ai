# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-01-05

### Added

- **Bitwise Logic Discovery**: Detection of `XOR` (`^`), `AND` (`&`), `OR` (`|`), `LSHIFT` (`<<`), `RSHIFT` (`>>`) patterns.
- **Offset Parametric Heuristic**: Discovery of `pi(x)+1`, `gamma(x)+1` variants.
- **Special Functions**: Added `prime_pi` (Prime Counting), `Gamma`, and `Bessel` function discovery.
- **Robustness**: Added guard clauses for complex numbers in digital logic and constant snapping for clean output.

### Fixed

- **Genetic Complexity**: Fixed bug where binary functions like `bitwise_xor` were treated as unary, leading to complexity underestimation.
- **Console Noise**: Silenced `ComplexWarning` logs during evolution.
- **Parser**: Fixed direct parsing of python bitwise operators (`<<`, `>>`, etc.) in REPL.

## [1.3.0] - 2025-12-19

### Added

- **Hybrid Evolution Mode**: `evolve --hybrid` seeds genetic algorithm with `find()` results for better convergence.
- **Boost Mode**: `evolve --boost N` multiplies population, generations, and timeout by N× (1-5).
- **Polyfit Fallback**: High-degree polynomial fitting (degrees 3,4,5) when R² < 0.95 from template methods.
- **Seed Protection**: 20% of seed copies kept unmutated to preserve good solutions.
- **Y-Normalization**: Automatic normalization for large value ranges (>1000) in evolve.
- **Implicit Multiplication**: Auto-converts `x(x+1)` to `x*(x+1)` with warning instead of error.

### Fixed

- **Evolve Timeout**: Pattern-based prevention of SymPy hangs (complexity > 50, nested powers).
- **Smart Seeding**: Fixed pole detection and priority in hint generation.

### Changed

- **Help Text**: Updated with new `--hybrid` and `--boost` flags.
- **Documentation**: Added high-degree polynomial limitations to README.

## [1.2.0] - 2025-12-11

### Added

- **LambertW Support**: Discovery of `exp(LambertW(log(x)))` pattern (Inverse of `x^x`).
- **Absolute Mode Documentation**: `README.md` and other docs rewritten for brevity.

### Fixed

- **REPL Parse Error**: Fixed syntax error during function persistence by separating confidence metadata from function string.
- **Version Banner**: CLI now correctly displays version number on startup.

## [1.1.0] - 2025-12-09

### Added

- **Function Persistence**: commands `save`, `loadfunction`, `clearfunction`, `clearsavefunction`.
- **Symbolic Constants**: Detection of `pi`, `e`, `sqrt(2)` in discovered formulas.
- **Basis Functions**: `x*log(x)` support.

### Fixed

- **CLI Parsing**: Fixed persistence command parsing conflict.
- **Physics Boosting**: Corrected relativistic interval detection.
- **Sphere Volume**: Fixed coefficient detection for `4/3*pi`.

## [1.0.2] - 2025-12-08

### Added

- **Shape Detection**: Softplus, Sigmoid, Tanh asymptotic detection.
- **Priority Override**: Solver prefers detected shapes over numerical fits.

### Fixed

- **Threshold Bug**: Corrected data size check (`<= 20`).
- **Feature Selection**: Fixed `log(1+exp(x))` vs `1/(340-x)` selection.

## [1.0.0] - 2025-12-06

### Added

- **Core Engine**: Symbolic Regression, SINDy, Genetic Programming.
- **Physics Detection**: Inverse Square, Euclidean, Geometric Mean, Harmonic Mean.
- **Calculator**: Solver, Calculus, Plotting.
- **Safety**: Sandboxed execution, resource limits.

### Fixed

- **Lazy Linear**: Prevented linear overfitting on sparse non-linear data.
- **Physics Patterns**: Added pre-checks for 2-variable laws.
