FROM nvcr.io/nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip python-is-python3

WORKDIR /rsrch

ENV POETRY_VERSION="1.5"
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install poetry==${POETRY_VERSION}

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project false && \
    poetry config virtualenvs.path /opt/poetry && \
    poetry config installer.max-workers 10

COPY ext ./ext

RUN --mount=type=cache,target=/root/.cache \
    poetry install --all-extras --with dev --with ext \
    --no-root --no-interaction --no-ansi && \
    poetry run poe torch-gpu

COPY .devcontainer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN apt-get update && \
    apt-get install -y git

ENTRYPOINT [ "/entrypoint.sh" ]