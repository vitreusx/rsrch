FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install APT packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y python3 python3-pip python3-venv curl git \ 
    build-essential

# Install Poetry
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    curl -sSL https://install.python-poetry.org --version 2.1.3 | python3 -

# Add `poetry` to the path
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install packages via Poetry
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-root
