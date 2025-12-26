# ==========================================
# 1. PHASE: BUILDER
# ==========================================
FROM python:3.10-slim as builder

# Prevent bytecode (.pyc) generation, print logs in real time.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (Some ML libraries require gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# We are creating a virtual environment (venv).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# 2. PHASE: RUNNER
# ==========================================
FROM python:3.10-slim as runner

WORKDIR /app

# SECURITY: Create a non-root user (appuser)
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy the ready-made virtual environment from the builder phase.
COPY --from=builder /opt/venv /opt/venv

# Environment settings
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy the project code.
COPY . .

# PERMISSION SETTING: Set the file owner to 'appuser'
RUN chown -R appuser:appgroup /app

# Switch to user (We are no longer root)
USER appuser

# Ports Used (For informational purposes)
# API: 8000, Streamlit: 8501, MLflow: 5000
EXPOSE 8000 8501 5000

# Start Command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]