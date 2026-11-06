"""Small filesystem boundaries shared by the desktop stores."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


def storage_path(root: Path, identifier: str, suffix: str = "") -> Path:
    """Resolve an opaque ID inside a store, rejecting traversal and external links."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", identifier):
        raise ValueError("Invalid storage identifier")
    if identifier.upper() in {"CON", "PRN", "AUX", "NUL", *(
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    )}:
        raise ValueError("Reserved storage identifier")
    root = root.resolve()
    target = (root / (identifier + suffix)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Storage path escapes its root")
    return target


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a file only after a complete write; temp names are unique per writer."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
