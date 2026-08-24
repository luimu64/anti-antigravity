# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    PORT=8000 \
    HOST=0.0.0.0 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

# curl (healthcheck) + Xvfb/x11vnc/websockify/novnc (one-time headless login)
# + Chromium runtime deps per playwright --with-deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    xvfb x11vnc websockify novnc \
    chromium \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/chromium /opt/ms-playwright-chromium

COPY --from=builder /root/.local /root/.local

# Copy application files
COPY pyproject.toml .
COPY app app/
COPY scripts scripts/
COPY main.py .

# Create data directory for credentials persistence
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["python", "main.py"]
