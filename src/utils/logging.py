from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(level: str = "INFO", *, log_path: Optional[str] = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    handlers.append(stream)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=lvl, handlers=handlers, force=True)

