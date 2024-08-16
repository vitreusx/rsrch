import re
from pathlib import Path


def sanitize(s: str):
    """Sanitizes a string for use in file paths, replacing prohibited characters by dashes."""
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", s)
