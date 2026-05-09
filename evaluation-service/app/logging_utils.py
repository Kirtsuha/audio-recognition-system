import logging
import os
import time
from contextlib import contextmanager


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str, **kwargs):
    started = time.perf_counter()
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)

    logger.info("Stage started: %s %s", stage_name, extra)

    try:
        yield
        elapsed = time.perf_counter() - started
        logger.info("Stage finished: %s elapsed_sec=%.2f", stage_name, elapsed)
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("Stage failed: %s elapsed_sec=%.2f", stage_name, elapsed)
        raise