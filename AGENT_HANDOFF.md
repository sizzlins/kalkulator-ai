# Agent Handoff: Kalkulator Development Guide

## Project Overview
Kalkulator is an AI-powered symbolic regression engine that discovers mathematical functions from data points. The goal is to create an "agentic learner" that discovers patterns from data, not from hardcoded physics knowledge.

## Core Philosophy: The Child Metaphor
We think of the engine as a **child learning to recognize patterns**:
- **Curiosity**: Detect power laws via Log-Log slope analysis
- **Ear for Music**: Detect frequencies via Zero-Crossing analysis
- **Humility**: Report R² confidence when uncertain
- **Transparency**: Show alternatives with warnings

## Key Files
| File | Purpose |
|------|---------|
| `kalkulator_pkg/regression_solver.py` | Main solver with OMP/BFSS selection |
| `kalkulator_pkg/function_finder_advanced.py` | Feature generation + detectors |
| `tests/test_user_failures.py` | Critical regression tests |

## Detection Functions (Phase 4)
```python
detect_power_laws(x, y)  # Returns candidate exponents from log-log slopes
detect_frequency(x, y)   # Returns candidate frequencies from zero-crossings
detect_curvature(x, y)   # Returns {'exp': k, 'poly': 2, 'log': True}
detect_saturation(x, y)  # Returns {'softplus': True, 'sigmoid': True, 'tanh': True}
```

## Current Architecture
1. **Feature Generation**: Creates ~100 candidate features from input data
2. **Detection-Triggered Boost**: Features matching detected patterns get priority
3. **BFSS**: Brute-force subset search for small data (<= 20 points)
4. **OMP**: Orthogonal Matching Pursuit for larger data
5. **R² Confidence**: Appends warning if R² < 0.9

## Known Limitation: Spurious Correlation
When a spurious feature (like `1/(340-x)`) numerically fits better than the detected pattern (like `log(1+exp(x))`), the engine may choose the wrong one. 

**Solution (from BotBicker debate)**: Implement transparent reporting:
- Primary result = Detected pattern
- Alternative = Numerical fit with warning

## Development Principles
1. **No Training Wheels**: Don't hardcode physics knowledge into boosts
2. **Detection, Not Prescription**: Let detectors find patterns, then boost matches
3. **Transparency Over Certainty**: Report alternatives with warnings
4. **Fail Gracefully**: Low R² should warn, not crash
5. **Root Cause First**: Never patch a bug with a quick fix. Always identify the fundamental flaw and refactor the architecture ("replace the wall") rather than applying band-aids.

## Test Commands
```bash
# Run all tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Run specific test
.venv\Scripts\python.exe -m pytest tests/test_user_failures.py -v
```

## Ideas Evaluated and Rejected
| Idea | Reason for Rejection |
|------|---------------------|
| Memory/caching of patterns | Conflicts with "agentic discovery" - creates bias toward cached solutions |
| Ensemble methods | Adds complexity without clear benefit; current override handles main cases |
| Fallback solver chain (OMP→GP) | GP is slow (seconds vs ms); already exists as explicit `evolve` command |
| Pattern confidence scoring | Already do per-instance quality checks (corr > 0.9); historical tracking adds complexity without benefit |

## Future Work (Vetted)
✅ **All identified improvements have been evaluated.** Current implementation is stable.

Potential enhancements (low priority):
- Verbose mode: `find f(x) --verbose` to show detection details
- Detection method in output: Show which detector triggered the result

## Recent Refinements (2025-12-08)

### Detection-Priority Override (with quality check)
When `detect_saturation` confirms sigmoid/softplus patterns, the solver force-selects the matching feature **only if correlation > 0.9**. This prevents override from forcing a bad fit.

### Residual-Based Hints
For really bad fits (R² < 0.7), analyze residuals and suggest missed patterns:
- `[Hint: try adding trig terms]` - if residuals show periodic patterns
- `[Hint: try sigmoid/softplus]` - if residuals show saturation

### Key Implementation Changes
- `regression_solver.py` line 260: Fixed `< 20` to `<= 20`
- `regression_solver.py` lines 279-306: Override with quality check (corr > 0.9)
- `regression_solver.py` lines 570-610: Residual hints for R² < 0.7 only

## Recent Refinements (2025-12-09)

### 1. Function Persistence
- Implemented `savefunction`, `loadfunction`, `clearfunction` commands.
- Functions are serialized to `~/.kalkulator_cache/functions.json`.
- This allows reusing complex function definitions across sessions and restarts.

### 2. Solver Architecture Refactor
- Monolithic `solver_legacy.py` refactored into `kalkulator_pkg/solver/` package.
- Specialized solvers: `algebraic.py`, `modular.py`, `system.py`, `inequality.py`.
- **Learnings**: When refactoring a core component that is imported everywhere, ensure `__init__.py` explicitly exports the original API to avoid breaking import paths.

### 3. Critical Fix: Floating-Point "Snap-to-Zero"
- **Issue**: `e^(i*pi)+1` resulted in `3.0e-30 - 2.2e-15*I` instead of `0`.
- **Fix**: Added epsilon check in `worker.py:_format_evaluation_result`.
- **Learning**: SymPy's `evalf` / `N()` inevitably introduces floating-point noise. ALWAYS implement a "snap-to-zero" threshold (e.g., `1e-9`) for user-facing output.

### 4. Critical Fix: Function Substitution
- **Issue**: `e^f(x)` where `f(x)=i*pi` evaluated incorrectly because `f(x)` was substituted as `i*pi`, yielding `e^i*pi` (interpreted as `(e^i)*pi`).
- **Fix**: `parser.py:expand_function_calls` now wraps substituted bodies in parentheses: `(i*pi)`.
- **Learning**: When substituting expressions textually or structurally, **ALWAYS** wrap in parentheses to preserve operator precedence.

## Engineering Standards (The Kalkulator Constitution)
Adapted from NASA/JPL's "Power of 10" rules for safety-critical systems:

1.  **Simple Control Flow**: Avoid complex "magic" (metaclasses, dynamic attribute injection, `exec`). Keep logic linear.
2.  **Bounded Loops**: All loops (especially in genetic programming and numerical solvers) must have a fixed `max_iterations` failsafe.
3.  **Object Stability**: Avoid runtime structure modification ("monkey patching"). Treat initialized objects as immutable where possible.
4.  **Small Units**: Functions should fit on a single screen (~60 lines). If larger, refactor.
5.  **Defensive Design**: Minimum 2 assertions per function. Validate assumptions (e.g., `assert x > 0`) to catch impossible states early.
6.  **Encapsulation**: Minimize global state. Keep variables scoped as locally as possible.
7.  **Explicit Error Handling**: Never swallow exceptions (`try: ... except: pass`). Handle failures explicitly or let them crash visibly.
8.  **No "Magic" Preprocessing**: Limit complex decorators that obscure function signatures. Code should be transparent.
9.  **Shallow Nesting**: Avoid deep nesting of `if/else/for`. Flatten logic to reduce cyclomatic complexity.
10. **Zero Tolerance for Warnings**: Treat strict `mypy` errors and `ruff` warnings as blocking bugs.
11. **Deep Assessment (The "Pondering" Rule)**: Before writing a single line of code, we MUST:
    - 100% understand the Root Cause (not just the symptom).
    - Plan the fix in detail.
    - "Double Think": Critically evaluate if the fix is actually a good idea or just a band-aid.
    - Only then, execute.
12. **Future-Proofing (The "Anti-Dev-Hell" Rule)**: Always design for the future. Ask: "Will this change make life miserable for the next developer?" If yes, don't do it. Avoid shortcuts that lead to technical debt or "development hell." Decouple concerns (e.g., calculation vs. presentation) to ensure long-term maintainability.

