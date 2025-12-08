# Kalkulator Architecture Documentation

## Module Dependencies

### Import Graph

```
kalkulator.py (entry point)
  └── kalkulator_pkg.cli.main_entry()

kalkulator_pkg/
├── api.py
│   ├── solver (solve_single_equation, solve_inequality, solve_system)
│   ├── worker (evaluate_safely)
│   └── parser, calculus, plotting
│
├── cli/ (Package)
│   ├── app.py
│   ├── commands/
│   │   └── debug.py
│   └── (Imports from solver, worker, etc.)
│
├── solver/ (Package)
│   ├── dispatch.py (solve_single_equation)
│   ├── system.py (solve_system)
│   ├── inequality.py (solve_inequality)
│   ├── algebraic.py
│   ├── numeric.py
│   └── worker (_worker_solve_cached, evaluate_safely)
│
├── worker.py
│   ├── config (constants)
│   ├── parser (parse_preprocessed)
│   └── types (ValidationError)
│   └── (NO IMPORTS FROM solver package - NO CIRCULAR DEPENDENCY ✅)
│
├── parser.py
│   ├── config (constants)
│   └── types (ValidationError)
│
└── types.py (standalone)
```

### Circular Import Analysis

**Status**: ✅ **No Circular Dependencies**

- `solver` package imports from `worker.py` (`_worker_solve_cached`, `evaluate_safely`)
- `worker.py` does **NOT** import from `solver` package
- Import direction: `solver` → `worker` (one-way dependency)

**Why This Works**:
- `worker.py` provides low-level evaluation and caching services
- `solver` uses these services but doesn't need to be imported by `worker.py`
- The dependency is intentionally one-way to avoid circular imports

**Future Considerations**:
- If `worker.py` ever needs solver functionality, consider:
  1. Moving shared code to a separate module (e.g., `shared.py`)
  2. Using dependency injection
  3. Refactoring to break the dependency

## Global State

### Current Implementation

**`_WORKER_MANAGER` Singleton** (in `worker.py`):

```python
_WORKER_MANAGER = _WorkerManager()
```

**Usage**:
- Used in `_worker_eval_cached()` and `_worker_solve_cached()`
- Accessed globally within `worker.py`
- Not passed as parameter to functions

**Configuration Modification** (in `cli/app.py`):

```python
# Direct module variable modification
_config.NUMERIC_FALLBACK_ENABLED = False
_config.OUTPUT_PRECISION = int(args.precision)
_config.CACHE_SIZE_PARSE = int(args.cache_size)
```

**Current Approach**:
- Direct module variable modification for simplicity
- Documented as "Future improvement: Use dependency injection"

### Future Improvements

**Option 1: Dependency Injection**
- Pass `WorkerManager` instance to functions that need it
- Pass configuration object to functions
- Eliminates global state

**Option 2: Configuration Object**
- Create `Config` class with all settings
- Pass `Config` instance to functions
- Allows multiple configurations in same process

**Option 3: Context Manager**
- Use context manager for temporary configuration changes
- Automatically restores original values

## Code Organization

### Module Responsibilities

#### `parser.py`
- Input preprocessing (sanitization, normalization)
- Expression parsing (string → SymPy)
- AST validation (security checks)
- Input limits enforcement

#### `worker.py`
- Sandboxed evaluation (isolated processes)
- Worker pool management
- Resource limiting (CPU, memory)
- Caching (evaluation, solving)
- Request cancellation

#### `solver/` (Package)
- Equation solving (dispatch to sub-modules)
- System solving (`system.py`)
- Inequality solving (`inequality.py`)
- Algebraic/Numeric strategies (`algebraic.py`, `numeric.py`)
- Solution formatting

#### `cli/` (Package)
- Command-line argument parsing (`app.py`)
- REPL logic (`app.py`, `context.py`)
- Command handlers (`commands/`)
- Output formatting
- Health checks

#### `api.py`
- Public Python API
- Type-safe return values (dataclasses)
- Error handling
- Documentation and examples

#### `config.py`
- Centralized configuration
- Environment variable support
- Constants and whitelists
- Version information

#### `types.py`
- Result dataclasses (`EvalResult`, `SolveResult`, `InequalityResult`)
- Exception types (`ValidationError`, `ParseError`, `SolverError`)

#### `calculus.py`
- Differentiation
- Integration
- Matrix operations (determinant)

#### `plotting.py`
- Function plotting (Matplotlib)
- ASCII plotting (fallback)

#### `logging_config.py`
- Structured logging setup
- Log format configuration
- Utility functions (`safe_log`)

## Data Flow

### Expression Evaluation

```
User Input → parser.preprocess()
           → parser.parse_preprocessed()
           → worker.evaluate_safely()
           → worker._worker_eval_cached()
           → _WORKER_MANAGER.request()
           → worker process
           → Result → User
```

### Equation Solving

```
User Input → parser.preprocess()
           → solver.solve_single_equation()
           → solver.evaluate_safely() (for LHS/RHS)
           → SymPy solving
           → Numeric fallback (if needed)
           → worker._worker_solve_cached() (for systems)
           → Result → cli.print_result_pretty()
           → User
```

## Concurrency Model

### Worker Pool Architecture

```
Main Process
  └── _WorkerManager (singleton)
       ├── Worker Process 1
       ├── Worker Process 2
       ├── Worker Process 3
       └── Worker Process 4 (default pool size)
       
       Each worker:
       - Has its own request queue
       - Shares response queue with manager
       - Has resource limits applied
       - Auto-restarts on failure
```

### Request Flow

1. Client calls `evaluate_safely()` or `solve_system()`
2. Cache check (`@lru_cache`)
3. If cache miss → `_WORKER_MANAGER.request()`
4. Round-robin routing to worker queue
5. Worker processes request
6. Response sent to shared response queue
7. Manager correlates response with request ID
8. Result returned to client

### Resource Limits (Unix Only)

Each worker process has:
- CPU time limit: 30 seconds (default)
- Memory limit: 400 MB (default)
- Applied via `resource.setrlimit()` (Unix only)

## Error Handling Strategy

### Error Types

1. **ValidationError**: Invalid input (forbidden tokens, too long, etc.)
2. **ParseError**: Expression parsing failure
3. **SolverError**: Equation solving failure
4. **TimeoutError**: Worker timeout
5. **TypeError/ValueError**: Unexpected type or value

### Error Propagation

- Internal functions return `{"ok": False, "error": "..."}` dictionaries
- Public API (`api.py`) converts to typed dataclasses with `ok=False`
- Exceptions logged with full traceback for unexpected errors
- User-facing errors are sanitized and context-aware

## Caching Strategy

### Cache Levels

1. **Parse Cache** (`@lru_cache` in `parser.py`)
   - Caches preprocessed → SymPy expressions
   - Size: 1024 entries (default)
   - In-memory only (process lifetime)

2. **Evaluation Cache** (`persistent` in `cache_manager.py`)
   - Caches preprocessed → evaluation results (JSON)
   - Persistent: Saved to `~/.kalkulator_cache/cache.json`
   - Survives process restarts
   - Size limit: 5000 entries (LRU eviction on save)

3. **Sub-expression Cache** (`persistent` in `cache_manager.py`)
   - Caches preprocessed sub-expressions → numeric values
   - Used for sub-expression replacement in preprocessing
   - Example: If `"2+2"` is cached as `"4"`, then `"(2+2)/2"` becomes `"4/2"` before parsing
   - Only caches pure numeric expressions (no variables)
   - Persistent: Saved to `~/.kalkulator_cache/cache.json`
   - Size limit: 10000 entries (LRU eviction on save)

4. **Solve Cache** (`@lru_cache` in `worker.py`)
   - Caches system solving results
   - Size: 512 entries (default)
   - In-memory only (process lifetime)

### Sub-Expression Caching

**How it works:**
1. When an expression like `"2+2"` is evaluated, if it's a pure numeric result, it's cached in the sub-expression cache.
2. When preprocessing a new expression like `"(2+2)/2"`, the parser scans for parenthesized sub-expressions.
3. If a sub-expression (e.g., `"2+2"`) is found in the cache, it's replaced with its cached value (`"4"`).
4. The resulting expression (`"4/2"`) is then parsed and evaluated, avoiding re-evaluation of the cached sub-expression.

**Benefits:**
- Faster evaluation of complex expressions with repeated sub-expressions
- Works across process restarts (persistent cache)
- Transparent to the user - no code changes needed

### Cache Invalidation

- Manual: `worker.clear_caches()` (called via REPL `clearcache` command) - clears both in-memory and persistent caches
- Automatic: LRU eviction when cache size exceeded (on save)
- Persistent cache is saved:
  - On normal program exit
  - On REPL quit (`quit` or Ctrl+C)
  - After each `--eval` command execution
- Cache location: `~/.kalkulator_cache/cache.json` (user home directory)

### Cache Loading

- Persistent cache is automatically loaded on startup (in `main_entry()` and `main()`)
- Cache version is checked - incompatible caches are automatically cleared
- If cache file is corrupted, an empty cache is used (graceful degradation)

## Security Architecture

### Defense in Depth

1. **Input Validation** (`parser.py`)
   - Denylist filtering
   - Length limits
   - Expression complexity limits

2. **AST Validation** (`parser.py`)
   - Tree traversal
   - Allowed node types only
   - No dangerous operations

3. **Process Isolation** (`worker.py`)
   - Separate worker processes
   - Resource limits (Unix)
   - Sandboxed evaluation

4. **Audit Logging** (`logging_config.py`)
   - Blocked input attempts
   - Resource violations
   - Security events

See `SECURITY.md` for complete security documentation.

---

**Last Updated**: 2025-11-02  
**Version**: 1.0.0
