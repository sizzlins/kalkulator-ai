# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Modulo operations**: Support for modulo equations (`x % 2 = 0`) with parametric solutions
- **Chinese Remainder Theorem**: Automatic solving of systems of congruences (e.g., `x = 1 % 2, x = 3 % 6, x = 3 % 7`)
- **Hexadecimal number parsing**: Automatic detection and conversion of hexadecimal numbers (both `0x` prefixed and hex-like strings)
- **Advanced function finding**: 
  - Extended polynomial/rational basis with inverse terms (`1/z`, `1/z^2`, `x*y/z^2`, etc.)
  - Automatic discovery of rational functions (e.g., Newton's gravitational law `x*y/z^2`)
  - Sparse solution search for finding simple explanations
  - Constant detection (π, e, sqrt(2), etc.) in function coefficients
  - High-precision parsing with Decimal and Fraction support
- **User-friendly error messages**: 
  - Clear, actionable error messages with helpful hints
  - Automatic detection of incomplete expressions, syntax errors, and common mistakes
  - Specific error messages for backslash continuation, invalid syntax, and parsing issues
- **None result handling**: Support for expressions that return `None` (e.g., `print()`) with graceful output handling
- **Improved REPL**: Better handling of multiple variable assignments and implicit multiplication (e.g., `xy=12, find x, y`)
- Comprehensive CONTRIBUTING.md with branch, commit, and PR rules
- Pre-commit hooks with black, isort, ruff, flake8, mypy
- pyproject.toml with centralized tool configuration
- Enhanced CI workflow with lint, type-check, test, and build jobs
- requirements-dev.txt with pinned development dependencies
- Conventional Commits message format enforcement

### Changed
- **Error handling**: Replaced generic parse errors with user-friendly messages and suggestions
- **Function finding**: Prioritizes extended basis with inverse terms for multi-parameter functions
- **Validation**: Allows `None` results for top-level expressions (e.g., side-effect functions like `print()`)
- Updated CI to use black line-length 88 (from 120)
- Standardized on ruff as primary linter (with flake8 fallback)
- Updated requirements.txt to remove dev dependencies

### Fixed
- **Backslash handling**: Improved error messages for expressions ending with backslash (line continuation)
- **Modulo equations**: Fixed validation errors when solving modulo equations
- **Hexadecimal parsing**: Fixed syntax errors when parsing hex-like numbers (leading zeros issue)
- **Duplicate error messages**: Removed redundant "Parse error: invalid syntax" messages when user-friendly messages are shown
- **REPL assignments**: Fixed handling of multiple assignments and `find` commands in REPL mode
- **None type validation**: Fixed `FORBIDDEN_TYPE` error when expressions return `None`
- Consistent code formatting across all modules
- Import ordering standardized

## [1.0.0] - 2025-01-XX

### Added
- Initial release of Kalkulator
- Core equation solving (linear, quadratic, polynomial)
- Pell equation detection and solving
- Inequality solving with chain support
- System of equations solving
- Calculus operations (differentiation, integration)
- Matrix determinant calculation
- ASCII and Matplotlib plotting support
- Sandboxed worker process execution
- Resource limits (CPU time, memory)
- Comprehensive input validation
- Security-focused expression parsing
- CLI with REPL mode
- Public Python API with typed returns
- Extensive test suite

### Security
- AST-based expression validation
- Whitelisted SymPy functions only
- Forbidden token detection
- Expression depth and node count limits
- Sandboxed worker processes with resource limits

## [0.1.0] - 2025-01-XX

### Added
- Initial development version
- Basic equation solving
- Core parser and solver modules

[Unreleased]: https://github.com/yourusername/kalkulator/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/kalkulator/releases/tag/v1.0.0
[0.1.0]: https://github.com/yourusername/kalkulator/releases/tag/v0.1.0

