import logging


class SkeletonUtils:

    @staticmethod
    def get_logger(log_file, verbose=False):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger('glow')
        logger.propagate = verbose
        logger.addHandler(logging.FileHandler(log_file))
        return logger
