FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "src.main"]
