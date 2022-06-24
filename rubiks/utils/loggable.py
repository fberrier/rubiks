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

    log_level = 'log_level'
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    name = 'name'

    def __init__(self, name=None, log_level=None):
        if name is None:
            name = self.__class__.__name__
        self.__init_name__ = name
        if log_level is None:
            log_level = __class__.INFO
        self.logger = getLogger(name)
        assert log_level in {__class__.DEBUG,
                             __class__.INFO,
                             __class__.WARNING,
                             __class__.ERROR,
                             }
        self.__init_log_level__ = log_level
        self.logger.setLevel({__class__.DEBUG: DEBUG,
                              __class__.INFO: INFO,
                              __class__.WARNING: WARNING,
                              __class__.ERROR: ERROR}.get(log_level, INFO))
        coloredlogs.install(level=log_level, logger=self.logger)
        self.initialised = False

    def do_init(self, log_level=None):
        """ Set or reset name = self.name() after fully initialised and if we ever actually log something """
        if self.initialised:
            return
        if hasattr(self, __class__.name):
            name = getattr(self, __class__.name)()
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
