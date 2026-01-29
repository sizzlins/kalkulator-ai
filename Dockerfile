# Kalkulator AI Worker Sandbox
# Secure, isolated environment for running symbolic regression worker processes.

FROM python:3.10-slim

# Create a non-root user for security
RUN groupadd -r kalkulator && useradd -r -g kalkulator -d /app -s /sbin/nologin -c "Kalkulator Worker" worker

# Install system dependencies (minimal)
# libgomp1 is often needed for numpy/scikit-learn optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Using a requirements export would be better, but listing key libs covers the worker needs
RUN pip install --no-cache-dir \
    numpy \
    sympy \
    scipy \
    scikit-learn \
    pandas

# Copy project package
COPY kalkulator_pkg /app/kalkulator_pkg
COPY worker_entrypoint.py /app/

# Set strict resource limits via env vars (can be overridden by docker-compose)
ENV WORKER_AS_MB=1024
ENV WORKER_CPU_SECONDS=5

# Drop to non-root user
USER worker

# Entrypoint to run the worker definition
CMD ["python", "-m", "kalkulator_pkg.worker"]
