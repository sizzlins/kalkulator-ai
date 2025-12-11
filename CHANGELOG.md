# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
