import logging
import time
from contextlib import contextmanager


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str, **kwargs):
    started = time.time()
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info("Stage started: %s %s", stage_name, extra)

    try:
        yield
        elapsed = time.time() - started
        logger.info("Stage finished: %s elapsed_sec=%.2f", stage_name, elapsed)
    except Exception:
        elapsed = time.time() - started
        logger.exception("Stage failed: %s elapsed_sec=%.2f", stage_name, elapsed)
        raise