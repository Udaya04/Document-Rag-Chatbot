"""Structured logging setup via loguru.

Removes loguru's default handler and installs a console handler (INFO) plus a
rotating file handler under logs/. The logs/ directory is created on import so
loguru can always write the file handler.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

os.makedirs("logs", exist_ok=True)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")

__all__ = ["logger"]
