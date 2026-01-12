# Function Finder Improvement Roadmap

## Status: Current Implementation Review

The current implementation has **9 working algorithms** but lacks several critical capabilities for production use. This document outlines the improvement plan.

---

## Immediate Priorities (Critical for Correctness)

### 1. ✅ PSLQ/nsimplify for Constant Detection
**Status**: Partially implemented (ad-hoc π detection)
**Priority**: HIGH
**Action Items**:
- [ ] Integrate SymPy's `nsimplify` with custom constant list
- [ ] Add PSLQ algorithm for integer relation detection
- [ ] Detect: π, e, log(2), sqrt(2), sqrt(3), etc.
- [ ] Replace numeric coefficients with symbolic constants when detected

**Implementation Plan**:
```python
def detect_symbolic_constants(coeff: Fraction | float) -> sp.Basic:
    """Detect if coefficient is close to a known constant."""
    # Use nsimplify with tolerance
    # Check against constant library
    # Return symbolic constant if match
```

---

### 2. ✅ High-Precision Parsing (Decimal/mpmath)
**Status**: NOT IMPLEMENTED
**Priority**: HIGH
**Action Items**:
- [ ] Add `Decimal` parsing for high-precision decimal inputs
- [ ] Integrate `mpmath` for arbitrary precision arithmetic
- [ ] Create unified parsing pipeline: int → Fraction → Decimal → mpmath
- [ ] Document precision policy

**Implementation Plan**:
```python
def parse_with_precision(val: str | float) -> Fraction | Decimal | mpf:
    """Parse with appropriate precision based on input format."""
    # Detect decimal places
    # Use Decimal for high-precision decimals
    # Use mpmath for very high precision
    # Convert to Fraction when possible
```

---

### 3. ✅ Explicit Tolerance Rules
**Status**: NOT IMPLEMENTED
**Priority**: HIGH
**Action Items**:
- [ ] Define absolute/relative tolerance constants
- [ ] Add tolerance parameters to all algorithms
- [ ] Implement residual threshold checking
- [ ] Add approximate vs exact fit flags

**Implementation Plan**:
```python
# config.py
ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-8
RESIDUAL_THRESHOLD = 1e-6

def is_exact_fit(computed, expected, abs_tol=ABSOLUTE_TOLERANCE, rel_tol=RELATIVE_TOLERANCE) -> bool:
    """Check if fit is exact within tolerances."""
```

---

### 4. ✅ Sparse Regression (LASSO/OMP)
**Status**: NOT IMPLEMENTED (uses exhaustive subset search)
**Priority**: HIGH
**Action Items**:
- [ ] Implement LASSO (L1-regularized) solver
- [ ] Implement Orthogonal Matching Pursuit (OMP)
- [ ] Replace exhaustive subset search for large problems
- [ ] Keep subset search as fallback with size limits

**Implementation Plan**:
```python
def sparse_regression_lasso(A, b, lambda_reg=0.01):
    """L1-regularized sparse regression."""
    # Use scipy.optimize or implement coordinate descent
    
def orthogonal_matching_pursuit(A, b, max_nonzero=None):
    """Greedy sparse solution."""
    # Iteratively select best matching columns
```

---

### 5. ✅ Model Selection (AIC/BIC/Cross-Validation)
**Status**: NOT IMPLEMENTED
**Priority**: HIGH
**Action Items**:
- [ ] Implement AIC (Akaike Information Criterion)
- [ ] Implement BIC (Bayesian Information Criterion)
- [ ] Add cross-validation for model comparison
- [ ] Prefer simpler models when multiple fits exist

**Implementation Plan**:
```python
def calculate_aic(n_params, n_samples, residuals):
    """AIC = 2k - 2ln(L) where k=params, L=likelihood."""
    
def calculate_bic(n_params, n_samples, residuals):
    """BIC = k*ln(n) - 2ln(L)."""
    
def select_best_model(candidates):
    """Select model with lowest AIC/BIC."""
```

---

## High Priority Improvements

### 6. Noise Handling and Statistical Validation
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM-HIGH
**Action Items**:
- [ ] Add χ² test for goodness of fit
- [ ] Calculate confidence intervals
- [ ] Report residuals table
- [ ] Distinguish exact vs approximate fits

---

### 7. Non-Polynomial Basis Library
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Add exponential basis: `exp(a*x)`
- [ ] Add logarithmic basis: `log(x)`
- [ ] Add trigonometric basis: `sin(x), cos(x)`
- [ ] Add rational functions: `1/(x+a)`
- [ ] Add Gaussian/RBF terms
- [ ] Make basis selection configurable

---

### 8. Symbolic Regression Engine
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Pattern library for common forms
- [ ] Genetic programming approach (optional)
- [ ] Template matching for known patterns
- [ ] Integration with external tools (optional)

---

### 9. Nonlinear Optimization
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Levenberg-Marquardt algorithm
- [ ] Global optimizers (differential evolution, etc.)
- [ ] Parameterized models: `a*sin(b*x+c)`
- [ ] Exact/symbolic post-processing

---

## Medium Priority Improvements

### 10. Integer/Diophantine Solvers
**Status**: PARTIAL (lcm-based integerization)
**Priority**: MEDIUM
**Action Items**:
- [ ] Specialized Diophantine solvers
- [ ] Quadratic/bilinear integer constraints
- [ ] Bounds propagation

---

### 11. Dimensional Analysis
**Status**: NOT IMPLEMENTED
**Priority**: LOW-MEDIUM
**Action Items**:
- [ ] Unit awareness
- [ ] Dimensional consistency checks
- [ ] Automatic scaling to reduce conditioning

---

### 12. Precision-Aware Interpolation Selection
**Status**: PARTIAL (has both methods)
**Priority**: MEDIUM
**Action Items**:
- [ ] Condition number checks
- [ ] Selection criteria (when to use Newton vs Lagrange)
- [ ] Fallback thresholds

---

### 13. Overfitting Detection
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Extrapolation warnings
- [ ] Bounded model options
- [ ] Detect polynomial divergence

---

### 14. Provenance and Reproducibility
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Deterministic monomial ordering
- [ ] Tie-breaking rules
- [ ] Metadata logging

---

## Lower Priority (But Important)

### 15. Performance Engineering
**Status**: NEEDS IMPROVEMENT
**Priority**: MEDIUM
**Action Items**:
- [ ] Greedy selection heuristics
- [ ] Randomized restarts
- [ ] Caching of monomial evaluations
- [ ] Sparse linear algebra
- [ ] Early stopping

---

### 16. User-Facing Diagnostics
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Residuals table output
- [ ] Rational-to-symbolic mapping log
- [ ] Exact vs approximate flags
- [ ] Suggested additional sample points

---

### 17. Input Sanitization
**Status**: PARTIAL
**Priority**: MEDIUM
**Action Items**:
- [ ] Explicit parsing policies
- [ ] Handle repeated x-values
- [ ] Floating-rounding provenance
- [ ] Scientific notation handling

---

### 18. External CAS Integration
**Status**: PARTIAL (uses SymPy)
**Priority**: LOW
**Action Items**:
- [ ] Optional hooks for advanced simplification
- [ ] Custom constant lists for nsimplify
- [ ] Rational reconstruction with limit_denominator

---

### 19. Security and Resource Limits
**Status**: NOT IMPLEMENTED
**Priority**: MEDIUM
**Action Items**:
- [ ] Time limits for combinatorial searches
- [ ] Memory caps
- [ ] Max subset size limits
- [ ] Early termination

---

### 20. Documentation and Extension Guide
**Status**: PARTIAL (has algorithm list)
**Priority**: LOW-MEDIUM
**Action Items**:
- [ ] Developer extension guide
- [ ] Decision-flow thresholds
- [ ] How to add new basis functions
- [ ] Monomial set extension guide

---

## Implementation Phases

### Phase 1: Critical Fixes (Week 1-2)
1. ✅ PSLQ/nsimplify constant detection
2. ✅ High-precision parsing
3. ✅ Explicit tolerance rules
4. ✅ Sparse regression (LASSO/OMP)
5. ✅ Model selection (AIC/BIC)

### Phase 2: Core Improvements (Week 3-4)
6. Noise handling and validation
7. Non-polynomial basis library
8. Performance engineering
9. Input sanitization

### Phase 3: Advanced Features (Week 5-6)
10. Symbolic regression
11. Nonlinear optimization
12. Diagnostics and explainability
13. Security and resource limits

### Phase 4: Polish (Week 7-8)
14. Documentation
15. Benchmarks and test coverage
16. Provenance and reproducibility
17. Dimensional analysis (optional)

---

## Testing Requirements

### New Test Suites Needed:
- [ ] High-precision parsing tests
- [ ] Constant detection tests (π, e, etc.)
- [ ] Tolerance and noise handling tests
- [ ] Sparse regression tests
- [ ] Model selection tests
- [ ] Performance benchmarks
- [ ] Pathological case suite

---

## Metrics for Success

1. **Correctness**: 100% test coverage for critical paths
2. **Performance**: Sub-second for typical problems (< 10 points)
3. **Robustness**: Handles noisy data, high-precision inputs
4. **Usability**: Clear diagnostics, exact vs approximate flags
5. **Extensibility**: Easy to add new basis functions

---

## References

- **PSLQ Algorithm**: Bailey, D. H., & Broadhurst, D. J. (2000). Parallel integer relation detection: techniques and applications.
- **LASSO**: Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
- **AIC/BIC**: Burnham, K. P., & Anderson, D. R. (2002). Model selection and multimodel inference.
- **OMP**: Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). Orthogonal matching pursuit.

