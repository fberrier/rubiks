########################################################################################################################
# Copyright (C) Francois Berrier 2020                                                                                  #
########################################################################################################################
import coloredlogs
from logging import getLogger, DEBUG, INFO, WARNING, ERROR
########################################################################################################################
from rubiks.utils.utils import pformat
########################################################################################################################


class Loggable:
    """ This class is a thin wrapper around Python's logger """

    def __init__(self, name=None, log_level=None):
        if name is None:
            name = self.__class__.__name__
        self.__init_name__ = name
        if log_level is None:
            log_level = 'INFO'
        self.logger = getLogger(name)
        assert log_level in {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
        self.__init_log_level__ = log_level
        self.logger.setLevel({'DEBUG': DEBUG, 'INFO': INFO, 'WARNING': WARNING, 'ERROR': ERROR}.get(log_level, INFO))
        coloredlogs.install(level=log_level, logger=self.logger)
        self.initialised = False

    def do_init(self, log_level=None):
        """ Set or reset name = self.name() after fully initialised and if we ever actually log something """
        if self.initialised:
            return
        if hasattr(self, 'name'):
            name = getattr(self, 'name')()
        else:
            name = self.__init_name__
        if log_level is None:
            log_level = self.__init_log_level__
        Loggable.__init__(self, name, log_level)
        self.initialised = True

    @staticmethod
    def process(arg):
        return pformat(arg)

    def log_debug(self, *args):
        self.logger.debug(self.format(*args))

    def log_info(self, *args):
        self.logger.info(self.format(*args))

    def log_warning(self, *args):
        self.logger.warning(self.format(*args))

    def log_error(self, *args):
        self.logger.error(self.format(*args))

    def format(self, *args):
        self.do_init()
        return ' '.join(str(self.process(arg)) for arg in args)

########################################################################################################################
