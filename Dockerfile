FROM python:3.13

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

WORKDIR /usr/src/app

COPY pyproject.toml uv.lock .python-version README.md /usr/src/app/
COPY src /usr/src/app/src

RUN uv sync --frozen --all-extras
