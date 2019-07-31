import logging


class SkeletonUtils:
    """Class to store auxiliar functions.

    Stores auxiliar project functionality as @staticmethod(s) instead of plain functions.
    """

    @staticmethod
    def get_logger():
        """Get a logger object to be used in the execution.

        Args:
            log_file: path to the log file where the logging is stored.
            verbose: whether or not to output log information through the terminal.

        Returns:
            An instace of a `logging.Logger` object properly configurated.
        """
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger('glow')

    @staticmethod
    def extend_logger(logger: logging.Logger, log_file: str, verbose=False):
        logger.propagate = verbose
        logger.addHandler(logging.FileHandler(log_file))
