# Release Notes

## v1.1.0 (2025-12-09) - Persistence Refactor & Physics Tuning

### ✨ New Features
- **Enhanced Function Persistence**: 
  - `clearfunction` now only clears functions from the *current session* (memory).
  - New `clearsavefunction` command completely wipes the saved function file from disk.
  - This separation prevents accidental data loss during experimentation.
- **Symbolic Constant Display**: The engine now detects and displays constants like `pi`, `4/3*pi`, `e`, and `sqrt(2)` finding formulas (e.g., Sphere Volume displays as `4/3*pi*r^3` instead of `4.188*r^3`).
- **New Basis Functions**: Added support for `x*log(x)` entropy-like terms.

### 🧪 Physics Tuning
- **Spacetime Interval Fix**: Adjusted structural boosting to correctly prefer pure squares for relativistic intervals (`-c^2t^2 + x^2 + y^2 + z^2`).
- **Sphere Volume Fix**: Corrected pre-check logic that was bypassing symbolic coefficient detection.

### 🐛 Bug Fixes
- Fixed CLI issue where persistence commands could be parsed as math expressions (added to `REPL_COMMANDS`).

---

## v1.0.2 (2025-12-08) - Agentic Discovery: Detection-Priority Override

### 🧠 Intelligent Pattern Detection
The engine now "trusts its pattern detector" over spurious numerical fits. When data exhibits sigmoid-family shapes (Softplus, Sigmoid, Tanh), the engine force-selects the detected pattern feature instead of being tricked by arbitrary fits like `1/(340-x)`.

### ✨ New Features
- **Detection-Priority Override**: Shape detectors (`detect_saturation`, `detect_curvature`) now directly influence feature selection
- **Softplus Discovery**: `log(1+exp(x))` correctly identified from asymptotic saturation patterns
- **Sigmoid Priority**: Double-saturation (both ends level off) treated as more specific than single-saturation

### 🐛 Bug Fixes
- Fixed threshold bug: `< 20` → `<= 20` to include 20-point datasets in detection block
- Fixed test failures for Softplus and Sigmoid discovery

### 📚 Documentation
- Added `agent_handoff.MD` - Development guide for future AI agents
- Documents the "Child Metaphor" philosophy for agentic discovery

---

## v1.0.1 (2025-12-07) - CI & Linting Fixes

### 🐛 Bug Fixes
- Fixed all Flake8 linting errors (F811 redefinition, F402 shadowing, E402 import order)
- Fixed all Ruff linting errors (E722 bare except, B006 mutable defaults, E741 ambiguous names)
- Fixed Mypy type errors (logger typing, cache_clear guard, python version)
- Resolved CI test hanging on Linux with pytest-timeout

### 🛠 Improvements
- Added `pytest-timeout==2.3.1` to prevent infinite test hangs
- CI now skips slow/performance tests for faster pipeline
- Each test has 120 second timeout limit
- Applied Black and isort formatting to all code

---

## v1.0.0 (2025-12-06) - Golden Master Stable

### 🚀 Major Milestone
Kalkulator AI has reached Version 1.0.0. This release marks the transition from Beta to a **feature-complete, stable scientific tool**.

### ✨ Key Features
- **Robust Symbolic Regression:** Discovers mathematical laws from data points (Polynomials, Rationals, Trigonometric, Exponentials).
- **Physics Law Detection:** Specialized handling for:
    - Inverse Square Laws (`m/r^2`)
    - Euclidean Distances (`sqrt(a^2+b^2)`)
    - Geometric Means (`sqrt(x*y)`)
    - Harmonic Means / Resistors (`xy/(x+y)`)
    - Relativistic Factors (`1/sqrt(1-v^2)`)
    - Difference of Squares (`a^2-b^2`)
- **Stability Guarantee:**
    - Safe Division handling for singularities (0/0, division by zero).
    - Complex root protection (negative domains masked).
- **Full Calculator Suite:**
    - Equation Solving (Linear, Quadratic, Systems).
    - Calculus (Differentiation, Integration).
    - Plotting (2D graphs via matplotlib).

### 🛠 Fixes & Improvements
- Fixed "Lazy Linear" shortcuts where the engine would prefer fitting a line to sparse data instead of the true curve.
- Implemented explicit pre-checks for 2-variable common physics patterns.
- Added "Golden Master" verification suite to prevent regression.

## 📦 Installation
```bash
pip install kalkulator
```
or from source:
```bash
git clone https://github.com/sizzlins/kalkulator-ai.git
cd kalkulator-ai
pip install .
```
