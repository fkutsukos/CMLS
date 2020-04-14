import logging
#~ LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s"
from colorlog import ColoredFormatter
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)


if __name__ == '__main__':


    try:
        cmd_split = str.split(raw_input('> : '), ' ')
    except KeyboardInterrupt:
        logger.info('Goodbye!')
        break