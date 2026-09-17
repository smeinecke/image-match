FROM python:3.9

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /usr/src/app

COPY pyproject.toml uv.lock .python-version README.md /usr/src/app/
COPY image_match /usr/src/app/image_match

RUN uv sync --frozen --all-extras
