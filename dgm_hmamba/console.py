"""Console helpers."""

import sys


def use_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so that symbols such as λ, ± and → print on
    Windows consoles, whose default code page (cp1252) cannot encode them."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - e.g. redirected streams in notebooks
            pass
