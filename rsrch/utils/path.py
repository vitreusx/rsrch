import re
from pathlib import Path


def sanitize(*parts: str | Path):
    """Sanitized a given path, replacing prohibited characters by dashes."""

    _parts = []
    for part in parts:
        for p in part.parts if isinstance(part, Path) else [part]:
            p = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", p)
            _parts.append(p)
    return Path(*_parts)
