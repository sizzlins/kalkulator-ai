# Worker Subprocess Fallback Documentation

## Purpose

The worker subprocess fallback is a safety mechanism that ensures the calculator can still function even when the persistent worker pool is unavailable or disabled.

## Architecture

The worker system has two execution paths:

1. **Persistent Worker Pool** (Primary): Uses `multiprocessing` to maintain a pool of worker processes that handle requests concurrently.

2. **Subprocess Fallback** (Secondary): If the persistent worker pool is unavailable or a request fails, the system falls back to spawning a subprocess for each request.

## When Fallback is Used

The subprocess fallback is triggered in the following scenarios:

1. **Persistent Worker Disabled**: When `ENABLE_PERSISTENT_WORKER=False` or `WORKER_POOL_SIZE=0`

2. **Worker Pool Unavailable**: When the persistent worker pool fails to start or becomes unresponsive

3. **Request Failure**: When a request to the persistent worker pool times out or returns an error

4. **Platform Limitations**: On systems where `multiprocessing` is not fully supported

## Implementation Details

### Primary Path (Persistent Pool)

```python
@lru_cache(maxsize=CACHE_SIZE_EVAL)
def _worker_eval_cached(preprocessed_expr: str) -> str:
    resp = _WORKER_MANAGER.request(
        {"type": "eval", "preprocessed": preprocessed_expr}, 
        timeout=WORKER_TIMEOUT
    )
    if isinstance(resp, dict):
        return json.dumps(resp)
    # Fallback to subprocess if request fails
    cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
    return proc.stdout or ""
```

### Fallback Path (Subprocess)

The fallback path:
1. Constructs a command to run the calculator in worker mode
2. Spawns a subprocess with the expression
3. Captures stdout (JSON response)
4. Returns the result

This ensures that even if the persistent pool fails, each evaluation gets its own isolated subprocess.

## Trade-offs

### Persistent Pool Advantages
- **Performance**: Processes are reused, avoiding startup overhead
- **Concurrency**: Multiple requests handled in parallel
- **Resource Efficiency**: Shared worker processes

### Subprocess Fallback Advantages
- **Reliability**: Works even when pool fails
- **Isolation**: Each request gets a fresh process
- **Simplicity**: No shared state concerns

### Subprocess Fallback Disadvantages
- **Performance**: Subprocess startup overhead per request
- **Resource Usage**: More memory/CPU per request (process creation)
- **Scalability**: Less efficient for high-throughput scenarios

## Configuration

Control worker mode via:

```bash
# CLI flag
kalkulator --worker-mode subprocess  # Force subprocess mode
kalkulator --worker-mode pool         # Use persistent pool (default)
kalkulator --worker-mode single       # Single worker process

# Environment variable
export KALKULATOR_ENABLE_PERSISTENT_WORKER=false  # Disable pool, force fallback
```

## Use Cases for Fallback

1. **Development/Debugging**: Easier to debug isolated subprocess execution
2. **Resource-Constrained Environments**: When process pool management is problematic
3. **Reliability Requirements**: When persistent pool failures are unacceptable
4. **Platform Compatibility**: On systems with limited multiprocessing support

## Testing

The fallback path is tested via:

1. Disabling persistent worker: `ENABLE_PERSISTENT_WORKER=False`
2. Simulating worker pool failures
3. Testing on platforms without full multiprocessing support

## Future Considerations

Potential improvements:
- Replace subprocess fallback with a more efficient mechanism
- Add retry logic with exponential backoff
- Implement graceful degradation strategies
- Add metrics/monitoring for fallback usage

## Conclusion

The subprocess fallback is a **critical safety mechanism** that ensures the calculator remains functional even when the primary worker pool is unavailable. While it has performance trade-offs, it provides essential reliability for production deployments.

**Recommendation**: Keep the fallback mechanism. It provides important resilience and is actively used when the persistent pool is disabled or unavailable.

