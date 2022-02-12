########################################################################################################################
# Copyright (C) Francois Berrier 2020                                                                                  #
########################################################################################################################
import coloredlogs
from logging import getLogger, DEBUG, INFO, WARNING, ERROR
########################################################################################################################


class Loggable:
    """ This class is a thin wrapper around Python's logger """

    def __init__(self, name: str, log_level: str='INFO'):
        self.logger = getLogger(name)
        assert log_level in {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
        self.logger.setLevel({'DEBUG': DEBUG, 'INFO': INFO, 'WARNING': WARNING, 'ERROR': ERROR}.get(log_level, INFO))
        coloredlogs.install(level=log_level, logger=self.logger)

    def log_debug(self, *args):
        self.logger.debug(self.format(*args))

    def log_info(self, *args):
        self.logger.info(self.format(*args))
        
    def log_warning(self, *args):
        self.logger.warning(self.format(*args))

    def log_error(self, *args):
        self.logger.error(self.format(*args))

    def format(self, *args):
        return ' '.join(str(self.process(arg)) for arg in args)

    def process(self, arg):
        """ TBD if need to process some stuff in a special manner """
        return arg

########################################################################################################################

