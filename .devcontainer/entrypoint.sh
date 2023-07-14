#!/bin/sh
. $(poetry env info -p)/bin/activate
poetry install --only-root --no-interaction --no-ansi
exec "$@"