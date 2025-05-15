import logging

def setup_logger(log_path='', default_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(default_level)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    return logger
