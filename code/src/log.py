import logging
import os
import sys

_ROOT = "deepscan"
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

_root_logger = logging.getLogger(_ROOT)
_root_logger.addHandler(_handler)
_root_logger.propagate = False


def configure(level: str = "info") -> None:
    """Set the root deepscan logger level.

    Accepted values: debug, info, warning, error, off (case-insensitive).
    Falls back to the DEEPSCAN_LOG_LEVEL env var, then 'info'.
    """
    if level is None:
        level = os.environ.get("DEEPSCAN_LOG_LEVEL", "info")
    level = level.strip().lower()
    if level == "off":
        _root_logger.setLevel(logging.CRITICAL + 1)
    else:
        numeric = getattr(logging, level.upper(), None)
        if numeric is None:
            raise ValueError(f"Unknown log level: {level!r}")
        _root_logger.setLevel(numeric)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the deepscan namespace."""
    return logging.getLogger(f"{_ROOT}.{name}")


# Apply default level at import time so the module is usable without
# an explicit configure() call (e.g. in unit tests or policy code).
configure(os.environ.get("DEEPSCAN_LOG_LEVEL", "info"))
