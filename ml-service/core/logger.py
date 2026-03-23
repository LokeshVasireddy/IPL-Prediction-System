import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_RETENTION_DAYS = 7


def get_logs_dir() -> Path:

    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir


def cleanup_old_logs(logs_dir: Path):

    cutoff = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)

    for file in logs_dir.glob("*.log"):

        file_time = datetime.fromtimestamp(file.stat().st_mtime)

        if file_time < cutoff:
            file.unlink()


def get_env():

    return os.getenv("ENV", "dev").lower()


def setup_logger(name: str):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    logs_dir = get_logs_dir()

    cleanup_old_logs(logs_dir)

    log_file = logs_dir / "ml_service.log"

    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=LOG_RETENTION_DAYS
    )

    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()

    if get_env() == "dev":
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
