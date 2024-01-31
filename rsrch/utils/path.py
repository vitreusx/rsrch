import re
from pathlib import Path


def sanitize(p: Path):
    """Sanitized a given path, replacing prohibited characters by dashes."""

    parts = []
    for part in p.parts:
        part = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", part)
        parts.append(part)
    return Path(*parts)
