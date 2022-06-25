########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentError, ArgumentParser
from abc import ABCMeta, abstractmethod
########################################################################################################################


class Parsable(metaclass=ABCMeta):
    """ Populates an ArgumentParser with the right fields. The way you do that is by
    implementing the populate_parser class method (which is abstract) and call
    add_argument as you wish.
     """

    default = 'default'
    action = 'action'
    store_true = 'store_true'

    __parsers__ = dict()
    __defaults_values__ = dict()
    parser_name = 'parser_name'

    @classmethod
    def default_values(cls):
        cls.setup_parser()
        return cls.__defaults_values__[cls]

    @staticmethod
    def get_all_parsable_parents(cls, parents):
        cls_parents = [parent for parent in cls.__bases__
                       if issubclass(parent, Parsable) and parent not in parents and parent is not Parsable]
        parents.extend(cls_parents)
        for parent in cls_parents:
            Parsable.get_all_parsable_parents(parent, parents)

    @classmethod
    def additional_dependencies(cls):
        return list()

    @classmethod
    def setup_parser(cls):
        if cls is Parsable:
            return
        if not issubclass(cls, Parsable):
            return
        if cls not in cls.__parsers__:
            # Create ArgumentParser for this class
            cls.__parsers__[cls] = ArgumentParser()
            # tag it
            setattr(cls.__parsers__[cls],
                    cls.parser_name,
                    cls.__name__)
            # go back to parents
            for parent in cls.__bases__:
                try:
                    parent.setup_parser()
                except AttributeError:
                    pass
            # populate the parser of this class with its own config first
            cls.populate_parser(cls.__parsers__[cls])
            # and then with that of parents, iteratively
            dependencies = list()
            cls.get_all_parsable_parents(cls, dependencies)
            # we also need for factories to take their widgets or additional dependencies
            other_dependencies = list()
            try:
                other_dependencies = cls.get_widgets()
            except AttributeError:
                pass
            try:
                other_dependencies += cls.additional_dependencies()
            except AttributeError:
                pass
            dependencies = other_dependencies + dependencies
            for dependency in dependencies:
                try:
                    dependency.populate_parser(cls.__parsers__[cls])
                except AttributeError:
                    pass
            # setup default values
            cls.__defaults_values__[cls] = vars(cls.__parsers__[cls].parse_args([]))
            # if widgets let us setup too
            for dependency in other_dependencies:
                try:
                    dependency.setup_parser()
                except AttributeError:
                    pass

    @classmethod
    def print_help(cls):
        cls.setup_parser()
        cls.__parsers__[cls].print_help()

    @classmethod
    def populate_parser(cls, parser):
        if cls is Parsable:
            return
        cls.populate_parser_impl(parser)

    @classmethod
    @abstractmethod
    def populate_parser_impl(cls, parser):
        return

    @classmethod
    def add_argument(cls, parser, field, **kw_args):
        try:
            if cls.action in kw_args and cls.store_true == kw_args[cls.action]:
                parser.add_argument('--%s' % field, **kw_args)
            else:
                parser.add_argument('-%s' % field, **kw_args)
        except ArgumentError:
            pass

    @classmethod
    def from_command_line(cls, command_line, strict=False):
        cls.setup_parser()
        if isinstance(command_line, str):
            command_line = command_line.strip().split(' ')
        if strict:
            return cls(**cls.__parsers__[cls].parse_args(command_line).__dict__)
        else:
            return cls(**cls.__parsers__[cls].parse_known_args(command_line)[0].__dict__)

    def __init__(self, **kw_args):
        self.setup_parser()
        for field, value in self.__defaults_values__[self.__class__].items():
            if hasattr(self, field):
                if getattr(self, field) != field:
                    continue
            setattr(self, field, kw_args.get(field, value))

    def get_config(self):
        self.setup_parser()
        return {field: getattr(self, field) for field in self.__class__.__defaults_values__[self.__class__].keys()}

########################################################################################################################

