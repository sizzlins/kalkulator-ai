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

## Test Commands
```bash
# Run all tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Run specific test
.venv\Scripts\python.exe -m pytest tests/test_user_failures.py -v
```

## Future Work
- [x] Implement transparent alternative reporting (Detection-Priority Override)
- [ ] Add residual-based feature synthesis (if R² low, analyze residual)
- [ ] Add memory/caching of successful patterns
- [ ] Consider ensemble methods for uncertain cases

## Recent Fix (2025-12-08)
**Detection-Priority Override**: When `detect_saturation` confirms sigmoid/softplus patterns, the solver now force-selects the matching feature over spurious numerical fits like `1/(340-x)`. This "trusts the detector" approach was validated by BotBicker debate.

Key changes:
- `regression_solver.py` lines 279-296: Override checks sigmoid first (more specific), then softplus
- `regression_solver.py` line 260: Fixed threshold from `< 20` to `<= 20` to include 20-point datasets
