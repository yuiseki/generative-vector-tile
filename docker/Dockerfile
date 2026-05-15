FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-c1v5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV HOST=0.0.0.0 \
    PORT=8080 \
    HOME=/tmp

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "python", "-m", "generative_vector_tile.server"]
