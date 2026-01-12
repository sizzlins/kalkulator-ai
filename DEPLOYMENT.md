# Deployment Guide

This guide covers deployment strategies for Kalkulator, including platform-specific considerations and containerization options.

## Installation

### Basic Installation

```bash
# Install dependencies
pip install sympy

# Optional dependencies for enhanced features
pip install numpy matplotlib  # For plotting features

# Run from source
python kalkulator.py
```

### Package Installation

```bash
# Install package
pip install -e .

# Use as command
kalkulator --version
kalkulator --health-check
```

## Platform Considerations

### Windows Deployment

**Important:** The `resource` module is Unix-only and not available on Windows. This means CPU and memory limits configured in `config.py` (e.g., `WORKER_CPU_SECONDS`, `WORKER_AS_MB`) will **not** be enforced on Windows systems.

#### Windows Deployment Recommendations

For production deployments on Windows with untrusted input, consider:

1. **Containerization (Recommended)**
   - Use Docker with resource limits configured
   - Example `docker-compose.yml`:
     ```yaml
     services:
       kalkulator:
         image: kalkulator:latest
         deploy:
           resources:
             limits:
               cpus: '0.5'
               memory: 400M
     ```

2. **Process Isolation**
   - Run worker processes under restricted user accounts with minimal privileges
   - Use Windows Job Objects API (requires additional implementation)
   - Monitor resource usage externally using Windows Performance Monitor

3. **Alternative Sandboxing**
   - Consider using a virtual machine or Windows Sandbox
   - Use AppContainers (Windows 10+)
   - Implement Windows-specific resource monitoring and termination

4. **Input Validation**
   - Rely more heavily on input validation and expression complexity limits
   - Monitor logs for suspicious patterns indicating resource exhaustion attempts

**Note:** For trusted environments (e.g., local development), Windows limitations are acceptable. The codebase gracefully handles the absence of the `resource` module.

### Unix/Linux Deployment

On Unix systems, resource limits are fully enforced:
- CPU time limits (`WORKER_CPU_SECONDS`)
- Memory limits (`WORKER_AS_MB`)
- Timeout limits (`WORKER_TIMEOUT`)

No additional configuration required beyond setting environment variables or using CLI flags.

### macOS Deployment

macOS supports resource limits similar to Linux, but some restrictions may apply depending on system configuration. Test resource limits in your specific environment.

## Containerization

### Dockerfile Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir sympy numpy matplotlib

# Copy application
COPY . /app

# Install package
RUN pip install -e .

# Set environment variables
ENV KALKULATOR_WORKER_CPU_SECONDS=30
ENV KALKULATOR_WORKER_AS_MB=400
ENV KALKULATOR_WORKER_TIMEOUT=60

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "from kalkulator_pkg.cli import main_entry; exit(0 if main_entry(['--health-check']) == 0 else 1)"

# Run application
ENTRYPOINT ["kalkulator"]
CMD []
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  kalkulator:
    build: .
    environment:
      - KALKULATOR_WORKER_CPU_SECONDS=30
      - KALKULATOR_WORKER_AS_MB=400
      - KALKULATOR_WORKER_TIMEOUT=60
      - KALKULATOR_LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 400M
        reservations:
          memory: 200M
    healthcheck:
      test: ["CMD", "kalkulator", "--health-check"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s
```

## Environment Variables

All configuration can be overridden via environment variables (prefixed with `KALKULATOR_`):

```bash
# Resource limits
export KALKULATOR_WORKER_CPU_SECONDS=30
export KALKULATOR_WORKER_AS_MB=400
export KALKULATOR_WORKER_TIMEOUT=60
export KALKULATOR_WORKER_POOL_SIZE=4

# Solver configuration
export KALKULATOR_NUMERIC_FALLBACK_ENABLED=true
export KALKULATOR_OUTPUT_PRECISION=6
export KALKULATOR_SOLVER_METHOD=auto

# Input validation limits
export KALKULATOR_MAX_INPUT_LENGTH=10000
export KALKULATOR_MAX_EXPRESSION_DEPTH=100
export KALKULATOR_MAX_EXPRESSION_NODES=5000

# Cache configuration
export KALKULATOR_CACHE_SIZE_PARSE=1024
export KALKULATOR_CACHE_SIZE_EVAL=2048
export KALKULATOR_CACHE_SIZE_SOLVE=256

# Numeric solver configuration
export KALKULATOR_MAX_NSOLVE_GUESSES=50
export KALKULATOR_NUMERIC_TOLERANCE=1e-8
export KALKULATOR_ROOT_SEARCH_TOLERANCE=1e-12
export KALKULATOR_MAX_NSOLVE_STEPS=80
export KALKULATOR_COARSE_GRID_MIN_SIZE=12
export KALKULATOR_ROOT_DEDUP_TOLERANCE=1e-6

# Logging
export KALKULATOR_LOG_LEVEL=INFO
export KALKULATOR_LOG_FILE=/var/log/kalkulator.log
```

## Production Checklist

- [ ] Review and set appropriate resource limits for your workload
- [ ] Configure logging (file logging recommended for production)
- [ ] Set up monitoring/alerting for resource usage
- [ ] Test worker pool size for optimal performance
- [ ] Verify health checks are accessible
- [ ] Review security settings (input validation limits)
- [ ] Test on target platform (Windows/Unix differences)
- [ ] Set up backup/recovery procedures
- [ ] Document deployment-specific configurations

## Health Checks

The application includes a built-in health check:

```bash
# CLI health check
kalkulator --health-check

# Programmatic check
python -c "from kalkulator_pkg.cli import main_entry; exit(main_entry(['--health-check']))"
```

Health check verifies:
- SymPy import and version
- Basic parsing functionality
- Basic solving functionality
- Worker evaluation (if available)
- Optional dependencies (NumPy, Matplotlib)

## Monitoring

### Recommended Metrics

- Worker process health (CPU, memory usage)
- Request success/failure rates
- Average request processing time
- Cache hit rates
- Error rates by error code
- Resource limit violations

### Logging

Configure logging via CLI or environment variables:

```bash
# CLI logging
kalkulator --log-level DEBUG --log-file kalkulator.log

# Environment variable logging
export KALKULATOR_LOG_LEVEL=INFO
export KALKULATOR_LOG_FILE=/var/log/kalkulator.log
```

Logs include:
- Request/response details
- Error traces
- Security violations (forbidden tokens/functions)
- Resource limit violations

## Troubleshooting

### Common Issues

1. **Worker timeout errors**
   - Increase `WORKER_TIMEOUT` or `KALKULATOR_WORKER_TIMEOUT`
   - Check system resource availability
   - Review expression complexity limits

2. **Memory issues**
   - Reduce `WORKER_POOL_SIZE`
   - Decrease cache sizes
   - Review `MAX_EXPRESSION_NODES` limit

3. **Windows resource limits not working**
   - Expected behavior (resource module unavailable)
   - Use containerization or alternative sandboxing
   - See Windows deployment recommendations above

4. **Import errors**
   - Verify all dependencies installed: `pip install sympy`
   - Check Python version (3.8+ required)
   - Run health check: `kalkulator --health-check`

## Security Considerations

- Review `SECURITY.md` for detailed security information
- Ensure input validation limits are appropriate for your threat model
- Monitor logs for suspicious patterns
- Use resource limits (Unix) or containerization (Windows)
- Keep dependencies updated

## Additional Resources

- `README.md` - Main project documentation
- `SECURITY.md` - Security considerations and mitigations
- `CONTRIBUTING.md` - Development guidelines
- `ARCHITECTURE.md` - Technical architecture details
- `CHANGELOG.md` - Version history
