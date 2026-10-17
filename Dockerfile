# syntax=docker/dockerfile:1
# Two stages: build the venv with uv (cached layers), then copy only the venv + code into a slim runtime image.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first (changes rarely → cached), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
# non-root user; the app only needs to write to /data (a volume)
RUN useradd --create-home --uid 10001 stridedj && mkdir /data && chown stridedj /data
COPY --from=build --chown=stridedj /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STRIDEDJ_DATA_DIR=/data
USER stridedj
WORKDIR /app
# long polling: outbound connections only, so no EXPOSE
ENTRYPOINT ["stridedj"]
CMD ["bot"]
