# Function-Finding Algorithms in Kalkulator

This document lists all algorithms implemented in Kalkulator to find functions from data points or unknown equations.

## Summary

Kalkulator implements **9 distinct algorithms** for finding functions from data points:

1. **fit_linear_from_two_points** - Exact linear fitting from 2 points
2. **Polynomial Basis Expansion (Single Parameter)** - Non-linear relationships for f(x)
3. **Polynomial Basis Expansion (Multi Parameter)** - Non-linear relationships for f(x,y,...)
4. **Lagrange Interpolation** - Polynomial interpolation (closed-form)
5. **Newton Divided Differences** - Numerically stable polynomial interpolation
6. **Vandermonde Matrix** - Fallback polynomial interpolation
7. **Linear Regression (Least Squares)** - Overdetermined/underdetermined systems
8. **Rational Gaussian Elimination** - Exact solving with Fraction arithmetic
9. **Edge Case Handlers** - Special scenarios (constants, zeros, etc.)

---

## Algorithm Details

### 1. fit_linear_from_two_points

**Purpose**: Find exact linear function f(x) = a*x + b from exactly 2 points using exact rational arithmetic.

**Algorithm**:
- Uses `Fraction` arithmetic throughout
- Computes slope: `a = (y2 - y1) / (x2 - x1)`
- Computes intercept: `b = y1 - a * x1`
- Returns multiple equivalent forms (standard, integer, alternative)
- Verifies solution by substituting both points

**When Used**: Explicitly called for 2-point linear fitting with exact results.

**Test Case**:
```python
f(5) = 10, f(15) = 3
Result: f(x) = -7/10*x + 27/2
        f(x) = (135 - 7*x) / 10
        f(x) = 27/2 - 7/10*x
```

**Location**: `kalkulator_pkg/function_manager.py:2375`

---

### 2. Polynomial Basis Expansion (Single Parameter)

**Purpose**: Find polynomial functions f(x) up to degree 2 using exact rational arithmetic.

**Algorithm**:
- Generates all monomials up to degree 2: `[1, x, x^2]`
- Builds exact `Fraction` matrix M where each row is `[1, x_i, x_i^2]`
- Solves `M * coeffs = outputs` using rational Gaussian elimination
- For underdetermined systems, searches for sparse solutions
- Automatically detects and simplifies π coefficients

**When Used**: Primary method for single-parameter functions with 1-3 data points.

**Test Cases**:
- `f(x) = x^2`: `[([0], 0), ([1], 1), ([2], 4)]` → `x**2`
- `f(x) = 2x + 1`: `[([0], 1), ([1], 3)]` → `2*x + 1`
- Constant: `[([10], 5)]` → `5`

**Location**: `kalkulator_pkg/function_manager.py:644` (`_find_sparse_polynomial_solution`)

---

### 3. Polynomial Basis Expansion (Multi Parameter)

**Purpose**: Find non-linear multi-parameter functions f(x,y,z,...) up to degree 2.

**Algorithm**:
- Generates all monomials up to degree 2: `[1, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2, ...]`
- Builds exact `Fraction` matrix with monomial evaluations
- Uses rational Gaussian elimination for exact solving
- For underdetermined systems, searches subsets for sparse solutions
- Prioritizes solutions with product terms

**When Used**: Primary method for multi-parameter functions.

**Test Cases**:
- `f(x,y) = x + y`: `[([0,0], 0), ([1,0], 1), ([0,1], 1)]` → `x + y`
- `f(x,y) = x*y`: `[([0,0], 0), ([1,0], 0), ([0,1], 0), ([2,3], 6)]` → `x*y`
- `f(x,y) = x^2 + y^2`: `[([0,0], 0), ([1,0], 1), ([0,1], 1), ([2,3], 13)]` → `x^2 + y^2`

**Location**: `kalkulator_pkg/function_manager.py:644` (`_find_sparse_polynomial_solution`)

---

### 4. Lagrange Interpolation

**Purpose**: Find polynomial of degree (n-1) that passes through n points exactly.

**Algorithm**:
- Builds Lagrange polynomial: `P(x) = Σᵢ yᵢ * Lᵢ(x)`
- Where `Lᵢ(x) = Πⱼ≠ᵢ (x - xⱼ)/(xᵢ - xⱼ)`
- Provides direct closed-form expression
- Automatically factors out common denominators

**When Used**: For single-parameter functions with 4+ points (when polynomial basis expansion doesn't find exact match).

**Test Cases**:
- Cubic: `[([0], 0), ([1], 0), ([2], 2), ([3], 12)]` → `x*(x^2 - 2*x + 1)`
- Quartic: `[([0], 0), ([1], 1), ([2], 16), ([3], 81), ([4], 256)]` → `x^4`

**Location**: `kalkulator_pkg/function_manager.py:1374-1478`

---

### 5. Newton Divided Differences

**Purpose**: Numerically stable polynomial interpolation (alternative to Lagrange).

**Algorithm**:
- Builds divided differences table
- Constructs Newton polynomial: `P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...`
- More numerically stable for high-degree polynomials
- Avoids Vandermonde conditioning issues

**When Used**: Fallback when Lagrange interpolation fails (especially for high-degree polynomials).

**Test Cases**:
- High-degree: `[([0], 0), ([1], 1), ([2], 32), ([3], 243), ([4], 1024), ([5], 3125)]` → `x^5`
- Closely spaced: `[([1.0], 1.0), ([1.1], 1.21), ([1.2], 1.44), ([1.3], 1.69)]` → `x^2`

**Location**: `kalkulator_pkg/function_manager.py:1483-1594`

---

### 6. Vandermonde Matrix

**Purpose**: Fallback polynomial interpolation using matrix inversion.

**Algorithm**:
- Builds Vandermonde matrix: each row is `[1, x, x², x³, ..., xⁿ⁻¹]`
- Solves `V * coeffs = outputs` by matrix inversion
- Checks condition number for numerical stability
- Warns about potential issues for high-degree polynomials

**When Used**: Final fallback when both Lagrange and Newton methods fail.

**Test Cases**:
- `[([-1], 1), ([0], 0), ([1], 1)]` → `x**2`

**Location**: `kalkulator_pkg/function_manager.py:1599-1763`

---

### 7. Linear Regression (Least Squares)

**Purpose**: Find best-fit linear function for overdetermined or underdetermined systems.

**Algorithm**:
- For overdetermined (m > n): Minimizes `||Ax - b||²`
- For underdetermined (m < n): Uses `x = A^T * (A * A^T)^(-1) * b` (minimum-norm solution)
- Handles cases where exact solution doesn't exist
- Uses SymPy for symbolic computation

**When Used**: 
- Overdetermined: More data points than parameters (best-fit)
- Underdetermined: Fewer data points than parameters (minimum-norm solution)

**Test Cases**:
- Overdetermined: `[([1,0], 2), ([0,1], 3), ([1,1], 5), ([2,0], 4)]` → `2*x + 3*y`
- Underdetermined: `[([1,1], 2)]` → Finds a solution or gives helpful error

**Location**: `kalkulator_pkg/function_manager.py:1980-2030`

---

### 8. Rational Gaussian Elimination

**Purpose**: Exact solving of linear systems using Fraction arithmetic.

**Algorithm**:
- Converts all inputs to exact `Fraction` objects
- Performs Gaussian elimination with exact rational arithmetic
- No floating-point errors
- Automatically reduces fractions

**When Used**: Internally by polynomial basis expansion and other exact methods.

**Test Cases**:
- Exact fractions: `[([1/2], 1/4), ([3/2], 9/4)]` → `x^2`
- Multi-param: `[([1/2, 1/3], 5/6), ([1/4, 1/5], 9/20)]` → Exact rational solution

**Location**: `kalkulator_pkg/function_manager.py:568` (`_gaussian_elim_rational`)

---

### 9. Edge Case Handlers

**Purpose**: Handle special scenarios gracefully.

**Scenarios**:
- **Single point**: Returns constant function
- **Zero function**: Detects and returns `0`
- **Three+ parameters**: Extends to arbitrary dimensions
- **Invalid inputs**: Provides helpful error messages

**Test Cases**:
- Constant: `[([5], 10)]` → `10`
- Zero: `[([0], 0), ([1], 0), ([2], 0)]` → `0`
- 3-param: `[([0,0,0], 0), ([1,0,0], 1), ([0,1,0], 1), ([0,0,1], 1)]` → `x + y + z`

**Location**: Throughout `kalkulator_pkg/function_manager.py`

---

## Algorithm Selection Flow

```
find_function_from_data(data_points, param_names):
  │
  ├─ Single point? → Return constant
  │
  ├─ Single parameter (f(x))?
  │   ├─ Try Polynomial Basis Expansion (degree 2)
  │   │   └─ Success? → Return result
  │   │
  │   └─ Try Polynomial Interpolation:
  │       ├─ Method 1: Lagrange Interpolation
  │       │   └─ Success? → Return result
  │       │
  │       ├─ Method 2: Newton Divided Differences
  │       │   └─ Success? → Return result
  │       │
  │       └─ Method 3: Vandermonde Matrix
  │           └─ Success? → Return result
  │
  └─ Multi-parameter (f(x,y,...))?
      ├─ Try Polynomial Basis Expansion (degree 2)
      │   └─ Success? → Return result
      │
      └─ Try Linear Regression (Least Squares)
          ├─ Overdetermined? → Best-fit solution
          └─ Underdetermined? → Minimum-norm solution or error
```

---

## Testing

All algorithms have been tested and verified working. Run the test suite:

```bash
python test_all_function_algorithms.py
```

**Test Results**: ✅ All 9 algorithms passing (100% success rate)

---

## Implementation Notes

1. **Exact Arithmetic**: All algorithms use `Fraction` or SymPy `Rational` for exact results
2. **Automatic Simplification**: Results are automatically simplified and formatted
3. **Error Handling**: Graceful fallbacks and helpful error messages
4. **Numerical Stability**: Newton method preferred for high-degree polynomials
5. **Sparse Solutions**: Polynomial basis expansion searches for simplest forms

---

## Usage Examples

### Via CLI:
```bash
python kalkulator.py
>>> f(5)=10, f(15)=3, find f(x)
>>> f(0,0)=0, f(1,0)=1, f(0,1)=1, find f(x,y)
```

### Via API:
```python
from kalkulator_pkg.function_manager import find_function_from_data, fit_linear_from_two_points

# Two-point linear fitting
a, b, (A, B, L), std, int_form, alt = fit_linear_from_two_points(5, 10, 15, 3)

# General function finding
success, func_str, factored, error = find_function_from_data(
    [([0], 0), ([1], 1), ([2], 4)],
    ["x"]
)
```

---

## References

- **Lagrange Interpolation**: [Wikipedia](https://en.wikipedia.org/wiki/Lagrange_polynomial)
- **Newton Divided Differences**: [Wikipedia](https://en.wikipedia.org/wiki/Newton_polynomial)
- **Vandermonde Matrix**: [Wikipedia](https://en.wikipedia.org/wiki/Vandermonde_matrix)
- **Least Squares**: [Wikipedia](https://en.wikipedia.org/wiki/Least_squares)
- **Gaussian Elimination**: [Wikipedia](https://en.wikipedia.org/wiki/Gaussian_elimination)

