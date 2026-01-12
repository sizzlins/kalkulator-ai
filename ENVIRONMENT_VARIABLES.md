# Environment Variables

Kalkulator supports configuration via environment variables. All environment variables are prefixed with `KALKULATOR_`.

## Resource Limits

### `KALKULATOR_WORKER_CPU_SECONDS`
- **Default**: `30`
- **Description**: Maximum CPU time (seconds) allowed per worker process
- **Example**: `export KALKULATOR_WORKER_CPU_SECONDS=60`

### `KALKULATOR_WORKER_AS_MB`
- **Default**: `400`
- **Description**: Maximum address space (MB) allowed per worker process
- **Example**: `export KALKULATOR_WORKER_AS_MB=512`

### `KALKULATOR_WORKER_TIMEOUT`
- **Default**: `60`
- **Description**: Timeout (seconds) for worker process operations
- **Example**: `export KALKULATOR_WORKER_TIMEOUT=120`

### `KALKULATOR_WORKER_POOL_SIZE`
- **Default**: `4`
- **Description**: Number of parallel worker processes in the pool
- **Example**: `export KALKULATOR_WORKER_POOL_SIZE=8`

### `KALKULATOR_ENABLE_PERSISTENT_WORKER`
- **Default**: `true`
- **Description**: Enable persistent worker pool (set to `false` to disable)
- **Example**: `export KALKULATOR_ENABLE_PERSISTENT_WORKER=false`

## Solver Configuration

### `KALKULATOR_NUMERIC_FALLBACK_ENABLED`
- **Default**: `true`
- **Description**: Enable numeric root-finding fallback (set to `false` to disable)
- **Example**: `export KALKULATOR_NUMERIC_FALLBACK_ENABLED=false`

### `KALKULATOR_SOLVER_METHOD`
- **Default**: `auto`
- **Description**: Solver method: `auto`, `symbolic`, or `numeric`
- **Example**: `export KALKULATOR_SOLVER_METHOD=symbolic`

### `KALKULATOR_MAX_NSOLVE_GUESSES`
- **Default**: `50`
- **Description**: Maximum number of guess points for numeric root finding
- **Example**: `export KALKULATOR_MAX_NSOLVE_GUESSES=100`

## Input Validation

### `KALKULATOR_MAX_INPUT_LENGTH`
- **Default**: `10000`
- **Description**: Maximum input string length (characters)
- **Example**: `export KALKULATOR_MAX_INPUT_LENGTH=50000`

### `KALKULATOR_MAX_EXPRESSION_DEPTH`
- **Default**: `100`
- **Description**: Maximum expression tree depth
- **Example**: `export KALKULATOR_MAX_EXPRESSION_DEPTH=200`

### `KALKULATOR_MAX_EXPRESSION_NODES`
- **Default**: `5000`
- **Description**: Maximum number of nodes in expression tree
- **Example**: `export KALKULATOR_MAX_EXPRESSION_NODES=10000`

## Output Configuration

### `KALKULATOR_OUTPUT_PRECISION`
- **Default**: `6`
- **Description**: Number of significant digits for numeric output
- **Example**: `export KALKULATOR_OUTPUT_PRECISION=10`

## Cache Configuration

### `KALKULATOR_CACHE_SIZE_PARSE`
- **Default**: `1024`
- **Description**: Size of parse cache (number of entries)
- **Example**: `export KALKULATOR_CACHE_SIZE_PARSE=2048`

### `KALKULATOR_CACHE_SIZE_EVAL`
- **Default**: `2048`
- **Description**: Size of evaluation cache (number of entries)
- **Example**: `export KALKULATOR_CACHE_SIZE_EVAL=4096`

### `KALKULATOR_CACHE_SIZE_SOLVE`
- **Default**: `256`
- **Description**: Size of solve cache (number of entries)
- **Example**: `export KALKULATOR_CACHE_SIZE_SOLVE=512`

## Usage Examples

### Set Multiple Variables (Unix/Linux/Mac)
```bash
export KALKULATOR_WORKER_TIMEOUT=120
export KALKULATOR_MAX_INPUT_LENGTH=20000
export KALKULATOR_OUTPUT_PRECISION=10
python -m kalkulator_pkg.cli
```

### Set Multiple Variables (Windows PowerShell)
```powershell
$env:KALKULATOR_WORKER_TIMEOUT="120"
$env:KALKULATOR_MAX_INPUT_LENGTH="20000"
$env:KALKULATOR_OUTPUT_PRECISION="10"
python -m kalkulator_pkg.cli
```

### Set Variables for Single Command (Unix/Linux/Mac)
```bash
KALKULATOR_WORKER_TIMEOUT=120 KALKULATOR_OUTPUT_PRECISION=10 python -m kalkulator_pkg.cli
```

## Priority

Configuration values are applied in the following priority (highest to lowest):

1. CLI flags (e.g., `--timeout`, `--precision`)
2. Environment variables (e.g., `KALKULATOR_WORKER_TIMEOUT`)
3. Default values in `config.py`

## Notes

- Boolean values: Use `true`/`false` (case-insensitive) or `1`/`0`
- Numeric values: Must be valid integers
- String values: Case-sensitive for choices like `KALKULATOR_SOLVER_METHOD`
- Changes to environment variables require restarting the application
- Some limits (like CPU/memory) are only enforced on Unix-like systems

