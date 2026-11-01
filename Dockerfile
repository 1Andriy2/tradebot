# Bot image (linux/amd64 and linux/arm64, e.g. Oracle Ampere). CPU-only PyTorch.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.6

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra ml --extra rl --extra db --no-dev
COPY src ./src
COPY config ./config
RUN uv sync --frozen --extra ml --extra rl --extra db --no-dev

RUN useradd --create-home bot && mkdir -p /app/data && chown bot /app/data
USER bot

ENTRYPOINT ["tradebot"]
CMD ["daemon"]
