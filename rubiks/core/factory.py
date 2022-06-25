########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
########################################################################################################################
from rubiks.core.parsable import Parsable
from rubiks.utils.utils import snake_case
########################################################################################################################


class Factory(Parsable, metaclass=ABCMeta):
    """ Factory base class.
     """

    __widget_types__ = dict()

    def __init__(self, **kw_args):
        Parsable.__init__(self, **kw_args)
        self.__populate__()
        setattr(self,
                self.factory_key_name(),
                snake_case(self.__class__.__name__))

    @classmethod
    def __populate__(cls, widget_types=None, force=False):
        if widget_types is None:
            widget_types = cls.widget_types()
        if cls not in cls.__widget_types__:
            cls.__widget_types__[cls] = dict()
        elif not force:
            return
        if isinstance(widget_types, (list, set)):
            widget_types = {snake_case(widget_type.__name__): widget_type for widget_type in widget_types}
        for widget_type_name, widget_type in widget_types.items():
            alias = snake_case(widget_type.__name__)
            cls.__widget_types__[cls][widget_type_name] = widget_type
            cls.__widget_types__[cls][alias] = widget_type

    @classmethod
    def widget_types(cls):
        """ Can overwrite to explicitly add some widgets that this factory supports """
        return list()

    @classmethod
    def get_widgets(cls):
        cls.__populate__()
        return list(cls.__widget_types__[cls].values())

    @classmethod
    def register_widget(cls, widget, widget_name=None):
        """ Or some widget can itself register """
        assert issubclass(widget, cls), '%s cannot register with factory %s' % (widget.__name__,
                                                                                cls.__name__)
        if cls not in cls.__widget_types__:
            cls.__widget_types__[cls] = dict()
        if widget_name is None:
            widget_name = snake_case(widget.__name__)
        cls.__widget_types__[cls][widget_name] = widget

    @classmethod
    @abstractmethod
    def factory_key_name(cls):
        return '%s_type' % snake_case(cls.__name__)

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.__populate__()
        cls.add_argument(parser,
                         field=cls.factory_key_name(),
                         type=str,
                         default=None)

    @classmethod
    def factory_type(cls, **kw_args):
        factory_key_name = cls.factory_key_name()
        if factory_key_name not in kw_args:
            raise ValueError('%s.factory expects mandatory argument %s' % (cls.__name__,
                                                                           factory_key_name))
        cls.__populate__()
        widget_types = cls.__widget_types__[cls]
        widget_name = kw_args[factory_key_name]
        if widget_name not in widget_types:
            raise ValueError('%s.factory unknown %s [%s]' % (cls.__name__,
                                                             factory_key_name,
                                                             widget_name))
        widget_type = widget_types[widget_name]
        assert issubclass(widget_type, cls),\
            'Factory.factory -> %s not a subclass of %s' % (widget_type.__name__,
                                                            cls.__name__)
        return widget_type

    @classmethod
    def factory(cls, **kw_args):
        factory_key_name = cls.factory_key_name()
        widget_name = kw_args[factory_key_name]
        widget_type = cls.factory_type(**kw_args)
        widget = widget_type(**kw_args)
        setattr(widget, factory_key_name, widget_name)
        return widget

    @classmethod
    def print_help(cls, **kw_args):
        if cls.factory_key_name() in kw_args:
            cls.factory(**kw_args).print_help()
        else:
            cls.setup_parser()
            cls.__parsers__[cls].print_help()
            print('Use print_help(%s=%s) for details' % (cls.factory_key_name(),
                                                         cls.factory_key_name()))
            widget_types = list()
            if cls in cls.__widget_types__:
                widget_types = list(cls.__widget_types__[cls].keys())
            print('\t%s in %s' % (cls.factory_key_name(), widget_types))

########################################################################################################################

