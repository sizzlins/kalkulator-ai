# Kalkulator API Documentation

## Overview

The Kalkulator API provides a clean, type-safe interface for mathematical operations. All functions return structured dataclasses rather than dictionaries, enabling static type checking and predictable interfaces.

## Installation

```python
# Install from package
from kalkulator_pkg import evaluate, solve_equation, solve_inequality
```

## Core Types

### EvalResult

Result of evaluating a mathematical expression.

```python
@dataclass
class EvalResult:
    ok: bool                          # Success indicator
    result: Optional[str] = None      # Exact result as string
    approx: Optional[str] = None      # Approximate numeric result
    free_symbols: Optional[List[str]] = None  # Variables in expression
    error: Optional[str] = None       # Error message if ok is False
```

### SolveResult

Result of solving an equation or system.

```python
@dataclass
class SolveResult:
    ok: bool                          # Success indicator
    result_type: str                 # "equation", "pell", "identity_or_contradiction", "multi_isolate", "system"
    error: Optional[str] = None       # Error message if ok is False
    exact: Optional[List[str]] = None  # Exact solutions (for equation type)
    approx: Optional[List[Optional[str]]] = None  # Approximate solutions
    solution: Optional[str] = None    # Solution string (for pell type)
    solutions: Optional[Dict[str, List[str]]] = None  # Multi-variable solutions
    system_solutions: Optional[List[Dict[str, str]]] = None  # System solutions
```

### InequalityResult

Result of solving an inequality.

```python
@dataclass
class InequalityResult:
    ok: bool                          # Success indicator
    result_type: str = "inequality"   # Always "inequality"
    error: Optional[str] = None       # Error message if ok is False
    solutions: Optional[Dict[str, str]] = None  # Solutions by variable
```

## API Functions

### Expression Evaluation

#### `evaluate(expression: str) -> EvalResult`

Evaluate a mathematical expression.

**Parameters**:
- `expression` (str): Mathematical expression string (e.g., "2+2", "sin(pi/2)", "x^2+1")

**Returns**: `EvalResult` with evaluation result

**Example**:
```python
from kalkulator_pkg import evaluate

result = evaluate("2 + 2 * 3")
assert result.ok is True
assert result.result == "8"
assert result.approx == "8.0"

result = evaluate("sin(pi/2)")
assert result.result == "1"

result = evaluate("x^2 + 1")
assert result.free_symbols == ["x"]
```

### Equation Solving

#### `solve_equation(equation: str, find_var: Optional[str] = None) -> SolveResult`

Solve a single equation.

**Parameters**:
- `equation` (str): Equation string (e.g., "x+1=0", "x^2-1=0")
- `find_var` (Optional[str]): Optional variable to solve for (e.g., "x")

**Returns**: `SolveResult` with solutions

**Example**:
```python
from kalkulator_pkg import solve_equation

# Linear equation
result = solve_equation("x + 1 = 0")
assert result.ok is True
assert result.result_type == "equation"
assert result.exact == ["-1"]
assert result.approx == ["-1.0"]

# Quadratic equation
result = solve_equation("x^2 - 1 = 0")
assert result.exact == ["-1", "1"]

# Pell equation
result = solve_equation("x^2 - 2*y^2 = 1")
assert result.result_type == "pell"
assert result.solution is not None
```

### Inequality Solving

#### `solve_inequality(inequality: str, find_var: Optional[str] = None) -> InequalityResult`

Solve an inequality.

**Parameters**:
- `inequality` (str): Inequality string (e.g., "x > 0", "1 < x < 5")
- `find_var` (Optional[str]): Optional variable to solve for

**Returns**: `InequalityResult` with solutions

**Example**:
```python
from kalkulator_pkg import solve_inequality

result = solve_inequality("x > 0")
assert result.ok is True
assert result.solutions["x"] is not None

result = solve_inequality("1 < x < 5")
assert result.ok is True
```

### System Solving

#### `solve_system(equations: str, find_var: Optional[str] = None) -> SolveResult`

Solve a system of equations.

**Parameters**:
- `equations` (str): Comma-separated equations (e.g., "x+y=3, x-y=1")
- `find_var` (Optional[str]): Optional variable to extract from solutions

**Returns**: `SolveResult` with system solutions

**Example**:
```python
from kalkulator_pkg import solve_system

result = solve_system("x+y=3, x-y=1")
assert result.ok is True
assert result.result_type == "system"
assert result.system_solutions is not None
# system_solutions: [{"x": "2", "y": "1"}]

# Extract specific variable
result = solve_system("x+y=3, x-y=1", find_var="x")
assert result.exact == ["2"]
```

### Validation

#### `validate_expression(expression: str) -> tuple[bool, Optional[str]]`

Validate an expression without evaluating it.

**Parameters**:
- `expression` (str): Expression string to validate

**Returns**: Tuple of `(is_valid, error_message)`

**Example**:
```python
from kalkulator_pkg import validate_expression

is_valid, error = validate_expression("x + 1")
assert is_valid is True

is_valid, error = validate_expression("__import__('os')")
assert is_valid is False
assert error is not None
```

## Calculus Operations

### `diff(expression: str, variable: Optional[str] = None) -> EvalResult`

Differentiate an expression.

**Parameters**:
- `expression` (str): Expression to differentiate (e.g., "x^3")
- `variable` (Optional[str]): Variable to differentiate with respect to (default: first variable found)

**Returns**: `EvalResult` with derivative

**Example**:
```python
from kalkulator_pkg import diff

result = diff("x^3")
assert result.result == "3*x**2"
```

### `integrate_expr(expression: str, variable: Optional[str] = None) -> EvalResult`

Integrate an expression.

**Parameters**:
- `expression` (str): Expression to integrate (e.g., "sin(x)")
- `variable` (Optional[str]): Variable to integrate with respect to

**Returns**: `EvalResult` with integral

**Example**:
```python
from kalkulator_pkg import integrate_expr

result = integrate_expr("x")
assert result.result == "x**2/2"
```

### `det(matrix_str: str) -> EvalResult`

Calculate matrix determinant.

**Parameters**:
- `matrix_str` (str): Matrix expression (e.g., "Matrix([[1,2],[3,4]])")

**Returns**: `EvalResult` with determinant

**Example**:
```python
from kalkulator_pkg import det

result = det("Matrix([[1,2],[3,4]])")
assert result.result == "-2"
```

## Plotting

### `plot(expression: str, variable: str = "x", x_min: float = -10, x_max: float = 10, ascii: bool = False) -> EvalResult`

Plot a single-variable function.

**Parameters**:
- `expression` (str): Function expression (e.g., "x^2")
- `variable` (str): Variable name (default: "x")
- `x_min` (float): Minimum x value (default: -10)
- `x_max` (float): Maximum x value (default: 10)
- `ascii` (bool): Return ASCII plot instead of opening window

**Returns**: `EvalResult` with plot data

**Example**:
```python
from kalkulator_pkg import plot

# ASCII plot
result = plot("x^2", ascii=True)
assert result.ok is True
print(result.result)  # Prints ASCII plot

# Matplotlib plot (opens window)
result = plot("sin(x)")
```

## Error Handling

### Exception Types

#### `ValidationError`
Raised when input validation fails.

```python
from kalkulator_pkg.types import ValidationError

try:
    evaluate("__import__('os')")
except ValidationError as e:
    print(e.message)  # Error message
    print(e.code)     # Error code
```

#### `ParseError`
Raised when parsing fails.

#### `SolverError`
Raised when solving fails.

### Error Codes

Common error codes:
- `TOO_LONG`: Input exceeds maximum length
- `TOO_COMPLEX`: Expression too complex (too many nodes)
- `TOO_DEEP`: Expression too deeply nested
- `FORBIDDEN_TOKEN`: Contains forbidden token
- `FORBIDDEN_FUNCTION`: Uses disallowed function
- `FORBIDDEN_TYPE`: Contains disallowed expression type
- `UNBALANCED_PARENS`: Unbalanced parentheses/brackets
- `VALIDATION_ERROR`: General validation error
- `PARSE_ERROR`: Parse error
- `SOLVER_ERROR`: Solver error

## Serialization

All result types support conversion to dictionaries:

```python
result = evaluate("2+2")
result_dict = result.to_dict()
# {"ok": True, "result": "4", "approx": "4.0", ...}
```

JSON serialization:

```python
import json
result = solve_equation("x+1=0")
json_str = json.dumps(result.to_dict())
```

## Best Practices

1. **Always check `ok` field**:
   ```python
   result = evaluate(expression)
   if not result.ok:
       handle_error(result.error)
   ```

2. **Use type hints**:
   ```python
   from kalkulator_pkg import EvalResult
   
   def process(result: EvalResult) -> None:
       ...
   ```

3. **Handle errors gracefully**:
   ```python
   try:
       result = solve_equation(equation)
   except ValidationError as e:
       logger.warning(f"Validation failed: {e}")
       return None
   ```

4. **Validate before expensive operations**:
   ```python
   is_valid, error = validate_expression(expr)
   if not is_valid:
       return early
   result = evaluate(expr)
   ```

## Configuration

Configuration is available via `kalkulator_pkg.config`:

```python
from kalkulator_pkg.config import (
    MAX_INPUT_LENGTH,
    WORKER_TIMEOUT,
    OUTPUT_PRECISION,
)
```

## Limitations

1. **SymPy Limitations**: Some operations may not be supported
2. **Resource Limits**: Large expressions may timeout
3. **Sandboxing**: Some legitimate operations may be blocked for security
4. **Numeric Precision**: Approximations may have limited precision

## Support

For issues or questions:
- GitHub Issues: [repository URL]
- Documentation: See README.md and ARCHITECTURE.md

