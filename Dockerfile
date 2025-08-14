FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN mkdir -p /app
WORKDIR /app

COPY ext ext
COPY poetry.lock pyproject.toml README.md ./
RUN mkdir rsrch && touch rsrch/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

COPY rsrch rsrch

ENTRYPOINT [ "python" ]