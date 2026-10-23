# The gateway. Build from the root of the repository:
#   docker build -f deploy/gateway.Dockerfile -t booking-gateway .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The locked dependencies without the development group, then the package itself.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

# What the pipeline reads at run time. The keys, the adapters and the logs are mounted.
COPY platforms ./platforms
COPY configs ./configs

RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p /app/logs \
    && chown app /app/logs
USER app

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]

CMD ["python", "-m", "adapt.gateway", "--host", "0.0.0.0", "--port", "8080"]
