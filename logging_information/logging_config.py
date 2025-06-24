import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE)]
)


def get_logger(name):
    """Create and return a logger with the given name."""
    return logging.getLogger(name)
