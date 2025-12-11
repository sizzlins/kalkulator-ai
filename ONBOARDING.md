# Onboarding Guide

## Setup

1. **Clone**:

   ```bash
   git clone <repository-url>
   cd Kalkulator
   ```

2. **Environment**:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Hooks**:
   ```bash
   pre-commit install
   ```

## Verification

1. **Health Check**:

   ```bash
   python -m kalkulator_pkg.cli --health-check
   ```

2. **Tests**:
   ```bash
   pytest tests/ -v
   ```

## Workflow

1. **Branch**: `git checkout -b feature/name main`
2. **Code**: Follow `CONTRIBUTING.md` standards.
3. **Verify**:
   ```bash
   black kalkulator_pkg tests
   isort kalkulator_pkg tests
   ruff check kalkulator_pkg tests
   mypy kalkulator_pkg
   pytest
   ```
4. **Commit**: Conventional format (`feat: ...`).
5. **PR**: Push and create Pull Request.

## Architecture

- **kalkulator_pkg/**: Core package.
  - `parser.py`: Input validation.
  - `solver.py`: Math engine.
  - `cli.py`: I/O handling.
  - `api.py`: Public interface.
- **tests/**: Test suite.

## Review Standards

- **Coverage**: ≥85%.
- **Style**: Black (88 chars), Google docstrings.
- **Types**: Strict checking.
- **Approval**: Required for merge.
