# Onboarding Guide for Kalkulator Contributors

Welcome to the Kalkulator project! This guide will help you get started contributing.

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Kalkulator
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before commits.

### 3. Verify Installation

```bash
python -m kalkulator_pkg.cli --health-check
```

This should show all checks passing.

### 4. Run Tests

```bash
pytest tests/ -v
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-feature main
   ```

2. **Make your changes** following the coding standards in CONTRIBUTING.md

3. **Run quality checks**:
   ```bash
   black kalkulator_pkg tests
   isort kalkulator_pkg tests
   ruff check kalkulator_pkg tests
   mypy kalkulator_pkg
   pytest
   ```

4. **Commit with conventional format**:
   ```bash
   git commit -m "feat(module): description of change"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/my-feature
   ```

### Code Standards

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 88 characters (Black default)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all public functions
- **Naming**: Descriptive names, no single-letter variables in public APIs

### Testing

- Write tests for all new features
- Maintain ≥85% code coverage
- Run `pytest --cov=kalkulator_pkg` to check coverage

## Project Structure

```
kalkulator_pkg/
├── config.py      # Configuration constants
├── parser.py      # Input parsing and validation
├── solver.py      # Equation/inequality solving
├── worker.py      # Sandboxed evaluation
├── cli.py         # Command-line interface
├── api.py         # Public Python API
├── types.py       # Result dataclasses
├── calculus.py    # Calculus operations
└── plotting.py    # Plotting functionality

tests/
├── test_parser.py
├── test_solver.py
├── test_integration.py
├── test_integration_cli.py
└── test_comprehensive.py
```

## Key Concepts

### Module Responsibilities

- **parser.py**: Pure functions for input preprocessing and parsing
- **solver.py**: Equation solving logic (returns dicts internally)
- **api.py**: Public API wrapper (returns typed dataclasses)
- **cli.py**: Thin wrapper that formats output and handles I/O

### Result Types

All public API functions return one of:
- `EvalResult`: Expression evaluation results
- `SolveResult`: Equation/system solving results
- `InequalityResult`: Inequality solving results

### Security Model

- Input validation at parser level
- AST-based expression validation
- Process isolation for evaluation
- Resource limits (CPU, memory)

## Common Tasks

### Adding a New Function

1. Implement in appropriate module
2. Add to `api.py` if it should be public
3. Add docstring with Args/Returns/Raises
4. Write unit tests
5. Update API.md documentation

### Fixing a Bug

1. Write a test that reproduces the bug
2. Fix the bug
3. Ensure test passes
4. Run full test suite
5. Update CHANGELOG.md

### Adding Configuration

1. Add constant to `config.py`
2. Document in module docstring
3. Add CLI flag if needed (in `cli.py`)
4. Update documentation

## Resources

- **CONTRIBUTING.md**: Detailed contribution guidelines
- **ARCHITECTURE.md**: Architecture documentation
- **API.md**: API reference
- **CHANGELOG.md**: Release history

## Getting Help

- Open a GitHub issue for bugs or questions
- Check existing issues and discussions
- Review code comments and docstrings
- Ask in discussions for guidance

## Code Review Process

1. All PRs require at least one approval
2. All CI checks must pass
3. Maintain ≥85% test coverage
4. Follow code style guidelines
5. Address all review comments

---

**Happy Contributing!** 🚀

