import logging
import colorlog

def setup_logger(log_path='', default_level='INFO'):
    if default_level.upper() == 'INFO':
        state = logging.INFO
    elif default_level.upper() == 'WARNING':
        state = logging.WARNING
    elif default_level.upper() == 'DEBUG':
        state = logging.DEBUG
    elif default_level.upper() == 'ERROR':
        state = logging.ERROR
    else:
        raise NotImplementedError(f'Invalid state: {default_level}...')

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(state)
    
    # File handler (no color)
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s', 
        datefmt=r'%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Color formatter for stream handler
    color_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(levelname)s] - %(message)s",
        datefmt=r'%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(color_formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    return logger
