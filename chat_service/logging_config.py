import logging
import sys

def setup_logging(log_level=logging.INFO):
    # Create logger
    logger = logging.getLogger('chat_service')
    logger.setLevel(log_level)

    # Prevent duplicate logs by removing existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger

# Create and configure logger
logger = setup_logging()
