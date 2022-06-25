########################################################################################################################
# Copyright (C) Francois Berrier 2020                                                                                  #
########################################################################################################################
import coloredlogs
from logging import getLogger, DEBUG, INFO, WARNING, ERROR
########################################################################################################################
from rubiks.core.parsable import Parsable
from rubiks.utils.utils import pformat
########################################################################################################################


class Loggable(Parsable):
    """ This class is a thin wrapper around Python's logger """

    log_level = 'log_level'
    debug = 'debug'
    info = 'info'
    warning = 'warning'
    error = 'error'
    known_log_levels = [debug, info, warning, error]
    name = 'name'
    get_name = 'get_name'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.log_level,
                         type=str,
                         choices=cls.known_log_levels,
                         default=cls.info)
        cls.add_argument(parser,
                         field=cls.name,
                         type=str,
                         default=None)

    def __init__(self, **kw_args):
        Parsable.__init__(self, **kw_args)
        if self.name is None or self.name == self.__class__.name:
            self.name = self.__class__.__name__
        self.logger = None
        self.setup_logger()
        self.re_named = False

    def setup_logger(self):
        self.logger = getLogger(self.name)
        level = {self.__class__.debug: DEBUG,
                 self.__class__.info: INFO,
                 self.__class__.warning: WARNING,
                 self.__class__.error: ERROR}.get(self.log_level)
        self.logger.setLevel(level)
        coloredlogs.install(level=self.log_level.upper(), logger=self.logger)

    def do_init(self):
        """ Set or reset name = self.name() after fully initialised and if we ever actually log something """
        if self.re_named:
            return
        if hasattr(self, __class__.get_name):
            if callable(self.get_name):
                self.name = getattr(self, __class__.get_name)()
        self.setup_logger()
        self.re_named = True

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
