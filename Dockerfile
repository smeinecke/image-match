FROM python:3.9

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY pyproject.toml uv.lock .python-version README.md /usr/src/app/
COPY image_match /usr/src/app/image_match

RUN uv sync --frozen --all-extras
