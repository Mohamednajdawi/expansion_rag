###############################################
# Stage 1: Build (The Builder)               #
###############################################
FROM python:3.13-slim AS builder

# 1. Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# 2. Install build dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
       build-essential \
       curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Leverage layer caching by copying dependency metadata first
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

# 4. Copy application source code
COPY src/ ./src/

# 5. Install Python dependencies into system Python
RUN uv pip install --system --no-cache-dir .

# 6. Remove build dependencies to keep image slim
RUN apt-get purge -y --auto-remove build-essential curl \
    && rm -rf /var/lib/apt/lists/*

###############################################
# Stage 2: Production (The Runner)           #
###############################################
FROM python:3.13-slim AS runner

# 1. Create a non-root user for security
RUN useradd --no-log-init --create-home appuser

WORKDIR /app

# 2. Copy installed packages and application code from builder
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

# 3. Ensure correct file permissions
RUN chown -R appuser:appuser /app
USER appuser

# 4. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 5. Expose port and define entrypoint
EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
