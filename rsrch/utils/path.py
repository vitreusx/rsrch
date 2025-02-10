import re


def sanitize(s: str, repl: str = "-"):
    """Sanitizes a string for use in file paths, replacing prohibited characters by `repl` (dashes by default.)"""
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", repl, s)
