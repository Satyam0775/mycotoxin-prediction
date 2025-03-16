# src/logging_config.py
import logging

def setup_logger():
    """ Set up a logger for runtime details. """
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
