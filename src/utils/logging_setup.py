"""One logging setup for every script: console plus a dated file in logs/."""

import logging
from datetime import datetime

from src.utils.paths import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and ``logs/<name>_<timestamp>.log``.

    Calling it twice with the same name reuses the existing handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"{name}_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
