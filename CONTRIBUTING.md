# Contributing to Kalkulator

Thank you for your interest in contributing to Kalkulator! This document outlines the project's development workflow, coding standards, and contribution guidelines.

## Development Workflow

### Branch Strategy

- **`main`**: Production-ready code. All merges to main require:
  - Passing CI checks (lint, type-check, tests)
  - At least one approving code review
  - Green test suite with ≥85% coverage

- **`refactor/*`**: Major refactoring branches
- **`feature/*`**: New feature branches
- **`fix/*`**: Bug fix branches
- **`docs/*`**: Documentation updates

### Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, CI, etc.)
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```
feat(solver): add Pell equation detection

Implemented specialized solver for Pell equations of the form x² - D*y² = 1.
Handles both positive and negative Pell equations with proper validation.

Fixes #42
```

```
fix(parser): handle Unicode superscripts in exponents

Previously, expressions like x² were not correctly converted to x**2.
Now properly maps Unicode superscript characters to numeric exponents.

Closes #38
```

### Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature main
   ```

2. **Make your changes** following the coding standards below

3. **Run local checks** before pushing:
   ```bash
   pre-commit run --all-files
   pytest
   mypy kalkulator_pkg
   ```

4. **Push and create a PR**:
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots/examples if applicable

5. **Ensure CI passes**:
   - All lint checks must pass
   - Type checking must pass
   - All tests must pass
   - Coverage must remain ≥85%

6. **Address review feedback**:
   - Make requested changes
   - Re-request review when ready
   - Keep PR focused (one logical change per PR)

7. **Merge approval**:
   - At least one maintainer approval required
   - All CI checks must be green
   - No unresolved review comments

## Coding Standards

### Code Style

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 88 characters (Black default)
- **Import order**: Automatic with `isort` (Black-compatible profile)
- **Type hints**: Required for all public functions and classes
- **Docstrings**: Required for all public functions, classes, and modules

### Running Code Quality Tools

```bash
# Format code
black kalkulator_pkg tests

# Sort imports
isort kalkulator_pkg tests

# Lint
ruff check kalkulator_pkg tests
# or
flake8 kalkulator_pkg tests

# Type check
mypy kalkulator_pkg

# Run tests
pytest

# Run with coverage
pytest --cov=kalkulator_pkg --cov-report=html
```

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Hooks will run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions
- **Test files**: `tests/test_<module>.py`

### Writing Tests

```python
import pytest
from kalkulator_pkg.solver import solve_single_equation

def test_solve_linear_equation():
    """Test solving a simple linear equation."""
    result = solve_single_equation("x + 1 = 0")
    assert result["ok"] is True
    assert result["exact"] == ["-1"]
```

### Test Requirements

- All public functions must have unit tests
- Edge cases must be covered
- Failure modes must be tested
- Integration tests for CLI workflows
- Maintain ≥85% code coverage

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_solver.py

# Specific test
pytest tests/test_solver.py::test_solve_linear_equation

# With coverage
pytest --cov=kalkulator_pkg --cov-report=term-missing

# Parallel execution
pytest -n auto
```

## Architecture Guidelines

### Module Responsibilities

- **`parser.py`**: Input parsing, preprocessing, validation (pure functions)
- **`solver.py`**: Equation/inequality/system solving (pure functions)
- **`cli.py`**: Command-line interface, formatting, I/O only
- **`api.py`**: Public Python API, typed dataclass returns
- **`worker.py`**: Sandboxed execution, resource limits
- **`config.py`**: Centralized configuration constants
- **`types.py`**: Shared dataclasses and exception types

### Design Principles

1. **Separation of concerns**: CLI, API, and core logic are separate
2. **Pure functions**: Core modules are importable without side effects
3. **Type safety**: Use typed dataclasses, not plain dicts
4. **Error handling**: Specific exceptions, not bare `except Exception:`
5. **Testing**: Core logic must be testable without CLI

### Adding New Features

1. Write tests first (TDD approach)
2. Implement in appropriate module
3. Add public API entry point if needed
4. Update documentation
5. Add example usage

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def solve_equation(eq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Solve a single equation.

    Args:
        eq_str: Equation string (e.g., "x+1=0", "x^2-1=0")
        find_var: Optional variable to solve for (e.g., "x")

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: Result type string
            - exact: List of exact solutions (strings)
            - approx: List of approximate solutions (strings or None)
            - error: Error message if ok is False

    Raises:
        ValidationError: If input validation fails
        ParseError: If parsing fails
    """
```

### Module Docstrings

Every module should start with:

```python
"""Brief one-line description.

This module handles:
- Feature 1 description
- Feature 2 description
- ...
"""
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Create release branch: `release/v<version>`
4. Run full test suite and CI
5. Create PR for review
6. Merge to `main`
7. Tag release: `git tag v<version>`
8. Push tag: `git push origin v<version>`

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code review**: Tag maintainers in your PR for feedback

## Code of Conduct

Be respectful and professional in all interactions. We welcome contributors of all backgrounds and experience levels.

---

Thank you for contributing to Kalkulator! 🚀
