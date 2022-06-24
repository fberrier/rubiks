########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentError
from abc import ABCMeta, abstractmethod
########################################################################################################################


class Parsable(metaclass=ABCMeta):
    """ Populates an ArgumentParser with the right fields. The way you do that is by
    implementing the populate_parser classmethod (which is abstract) and call
    add_argument as you wish.
     """

    default = 'default'

    @classmethod
    @abstractmethod
    def populate_parser(cls, parser):
        return

    @classmethod
    def add_argument(cls, parser, field, **kw_args):
        try:
            parser.add_argument('-%s' % field, **kw_args)
        except ArgumentError:
            pass

########################################################################################################################
