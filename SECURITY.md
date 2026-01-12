# Security Considerations

This document outlines security measures, limitations, and recommended deployment practices for Kalkulator.

> **⚠️ Windows Users**: Resource limits (CPU/memory) do not apply on Windows due to OS limitations. See the [Windows Resource Limits](#windows-resource-limits) section below for deployment recommendations.

## Security Measures

### Input Validation

1. **String-level filtering**: Denylist of dangerous tokens (`__`, `import`, `eval`, etc.)
2. **Preprocessing checks**: Input length limits (10,000 chars), balanced parentheses
3. **AST-level validation**: Expression tree traversal rejecting dangerous node types
4. **Whitelist-based parsing**: Only allowed SymPy functions can be parsed

### Sandboxing

- **Worker processes**: User input is parsed in isolated worker processes
- **Resource limits** (Unix only):
  - CPU time: 30 seconds per worker
  - Memory: 400 MB per worker
- **Windows limitation**: Resource limits don't apply on Windows (OS limitation)

### Expression Complexity Limits

- Maximum input length: 10,000 characters
- Maximum expression depth: 100 levels
- Maximum expression nodes: 5,000 nodes

## Known Limitations

### SymPy Parser

SymPy's `parse_expr()` uses Python's `eval` internally. While we:
- Restrict allowed names via `local_dict`
- Validate expression tree structure
- Use resource limits

A sophisticated attacker could potentially find ways to execute code. For production use with untrusted input, consider:
- Running in a container/virtual machine
- Additional OS-level sandboxing
- Process isolation with user privileges

### Windows Resource Limits

**Important**: The `resource` module is Unix-only and is not available on Windows. This means CPU and memory limits configured in `config.py` (e.g., `WORKER_CPU_SECONDS`, `WORKER_AS_MB`) will not be enforced on Windows systems.

#### Windows Deployment Recommendations

For production deployments on Windows with untrusted input, consider:

1. **Containerization (Recommended)**:
   - Use Docker with resource limits configured in `docker-compose.yml` or Kubernetes
   - Example Docker resource limits:
     ```yaml
     deploy:
       resources:
         limits:
           cpus: '0.5'
           memory: 400M
     ```

2. **Process Isolation**:
   - Run worker processes under restricted user accounts with minimal privileges
   - Use Windows Job Objects API (requires additional implementation)
   - Monitor resource usage externally using Windows Performance Monitor or similar tools

3. **Alternative Sandboxing**:
   - Consider using a virtual machine or Windows Sandbox
   - Use AppContainers (Windows 10+)
   - Implement Windows-specific resource monitoring and termination

4. **Input Validation**:
   - Rely more heavily on input validation and expression complexity limits
   - Monitor logs for suspicious patterns indicating resource exhaustion attempts

#### Implementation Status

The codebase gracefully handles the absence of the `resource` module:
- `HAS_RESOURCE` flag detects availability
- Worker processes start successfully on Windows without resource limits
- Error messages are clear when resource-related errors occur

**Security Impact**: On Windows, malicious input could potentially consume unlimited CPU/memory in worker processes. For trusted environments (e.g., local development), this is acceptable. For untrusted input, use one of the mitigation strategies above.

## Recommended Deployment Posture

### For Trusted Users

- Current implementation is sufficient
- Monitor logs for unusual patterns
- Regular dependency updates

### For Untrusted Input

1. **Containerization**: Run in Docker with resource limits
2. **Network isolation**: Disable network access for worker processes
3. **User privileges**: Run worker processes as unprivileged user
4. **Monitoring**: Log all parse failures and resource exhaustion
5. **Rate limiting**: Limit requests per user/IP

### Hardening Checklist

- [ ] Run workers in container/VM
- [ ] Use unprivileged user for workers
- [ ] Enable all logging options
- [ ] Monitor resource usage
- [ ] Pin dependency versions
- [ ] Regular security audits
- [ ] Set restrictive file system permissions

## Threat Model

### High-Risk Scenarios

1. **Code injection via SymPy**: Mitigated by whitelist + AST validation
2. **Resource exhaustion**: Mitigated by limits (Unix) and monitoring
3. **Information leakage**: Mitigated by sanitized error messages

### Attack Vectors

- **Malformed input**: Handled by validation layers
- **Complex expressions**: Limited by complexity constraints
- **Worker process escape**: Mitigated by isolation and limits

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:
1. Do not open a public issue
2. Contact the maintainer privately
3. Allow time for fix before disclosure

## Security Updates

- Regularly update dependencies (`pip audit`, `safety check`)
- Monitor SymPy security advisories
- Review and update whitelist as needed

