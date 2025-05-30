import logging
import os
from logging.handlers import RotatingFileHandler
from utils.config import LogConfig

def setup_logger():
    """
    Sets up the logger for the application.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LogConfig.LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(getattr(logging, LogConfig.LOG_LEVEL))

    # Create console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LogConfig.LOG_LEVEL))

    # Create file handler for logging to a file
    file_handler = RotatingFileHandler(
        LogConfig.LOG_FILE,
        maxBytes=LogConfig.MAX_LOG_SIZE,
        backupCount=LogConfig.BACKUP_COUNT
    )
    file_handler.setLevel(getattr(logging, LogConfig.LOG_LEVEL))

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create the logger instance
logger = setup_logger()
