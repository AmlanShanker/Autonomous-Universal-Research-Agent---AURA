# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.17 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install only locked production dependencies and the AURA project.
COPY pyproject.toml uv.lock README.md ./
COPY . .
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin aura \
    && chown -R aura:aura /app

USER aura

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
