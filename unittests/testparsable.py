########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.core.parsable import Parsable
from rubiks.core.factory import Factory
from rubiks.utils.utils import snake_case
########################################################################################################################


class SomeParsable(Parsable):
    some_string = 'some_string'
    some_int = 'some_int'
    some_bool = 'some_bool'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.some_string,
                         type=str,
                         choices=['what', 'is', 'this?'])
        cls.add_argument(parser,
                         field=cls.some_int,
                         type=int,
                         default=413)
        cls.add_argument(parser,
                         field=cls.some_bool,
                         default=False,
                         action=cls.store_true)

########################################################################################################################


class SomeDerivedParsable(SomeParsable):

    some_float = 'some_float'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.some_float,
                         type=float,
                         default=3.14)

########################################################################################################################


class SomeDerivedParsableFactory(SomeDerivedParsable, Factory):

    some_float = 'some_float'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.some_float,
                         type=float,
                         default=4.13)

    @classmethod
    def factory_key_name(cls):
        return snake_case('SomeDerivedParsableFactoryType')

########################################################################################################################


class Widget1(SomeDerivedParsableFactory):

    widget_1_config = 'widget_1_config'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.widget_1_config,
                         type=int,
                         nargs='+',
                         default=(1, 2, 3))

########################################################################################################################


class Widget2(SomeDerivedParsableFactory):

    widget_2_config = 'widget_2_config'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.widget_2_config,
                         action=cls.store_true,
                         default=False)

########################################################################################################################


class TestParsable(TestCase):

    def test_parsable(self):
        my_parsable = SomeParsable(some_string='this?')
        self.assertEqual('this?', my_parsable.some_string)
        self.assertEqual(413, my_parsable.some_int)
        self.assertFalse(my_parsable.some_bool)
        my_parsable2 = SomeParsable(some_string='what', some_bool=True)
        self.assertEqual('what', my_parsable2.some_string)
        self.assertEqual(413, my_parsable2.some_int)
        self.assertTrue(my_parsable2.some_bool)
        my_parsable3 = SomeParsable(some_string='is', some_bool=False)
        self.assertEqual('is', my_parsable3.some_string)
        self.assertEqual(413, my_parsable3.some_int)
        self.assertFalse(my_parsable3.some_bool)
        # the following should not crap out though
        my_parsable4 = SomeParsable(some_string='is', some_bool=False, some_thing=34)
        self.assertEqual('is', my_parsable4.some_string)
        self.assertEqual(413, my_parsable4.some_int)
        self.assertFalse(my_parsable4.some_bool)

    def test_parsable_from_command_line(self):
        command_line = '-some_thing=this? -some_int=1979'
        my_parsable = SomeParsable.from_command_line(command_line)
        self.assertEqual('this?', my_parsable.some_string)
        self.assertEqual(1979, my_parsable.some_int)
        self.assertFalse(my_parsable.some_bool)
        command_line += ' --some_bool'
        my_parsable = SomeParsable.from_command_line(command_line)
        self.assertEqual('this?', my_parsable.some_string)
        self.assertEqual(1979, my_parsable.some_int)
        self.assertTrue(my_parsable.some_bool)

    def test_parsable_from_command_line(self):
        my_parsable = SomeParsable.from_command_line(["-some_string=this?", "--some_bool"])
        self.assertEqual('this?', my_parsable.some_string)
        self.assertEqual(413, my_parsable.some_int)
        self.assertTrue(my_parsable.some_bool)
        my_parsable2 = SomeParsable.from_command_line("-some_string=this? --some_bool")
        self.assertEqual('this?', my_parsable2.some_string)
        self.assertEqual(413, my_parsable2.some_int)
        self.assertTrue(my_parsable2.some_bool)

    def test_derived_parsable(self):
        my_derived_parsable = SomeDerivedParsable.from_command_line("-some_string=this? --some_bool -some_float=1.3")
        self.assertEqual('this?', my_derived_parsable.some_string)
        self.assertEqual(413, my_derived_parsable.some_int)
        self.assertEqual(1.3, my_derived_parsable.some_float)
        self.assertTrue(my_derived_parsable.some_bool)
        my_derived_parsable2 = SomeDerivedParsable(some_string='what', some_bool=False)
        self.assertEqual('what', my_derived_parsable2.some_string)
        self.assertEqual(413, my_derived_parsable2.some_int)
        self.assertEqual(3.14, my_derived_parsable2.some_float)
        self.assertFalse(my_derived_parsable2.some_bool)

    def test_print_help_some_parsable(self):
        SomeParsable.print_help()
        self.assertEqual(SomeParsable.default_values()[SomeParsable.some_int],
                         413)

    def test_print_help_some_derived_parsable(self):
        SomeDerivedParsable.print_help()
        self.assertEqual(SomeDerivedParsable.default_values()[SomeDerivedParsable.some_int],
                         413)
        self.assertEqual(SomeDerivedParsable.default_values()[SomeDerivedParsable.some_float],
                         3.14)

    def test_print_help_some_derived_parable_factory(self):
        SomeDerivedParsableFactory.register_widget(Widget1)
        SomeDerivedParsableFactory.register_widget(Widget2)
        SomeDerivedParsableFactory.print_help()
        self.assertEqual(SomeDerivedParsableFactory.default_values()[SomeDerivedParsableFactory.some_int],
                         413)
        self.assertEqual(SomeDerivedParsableFactory.default_values()[SomeDerivedParsableFactory.some_float],
                         4.13)

    def test_print_help_some_derived_parable_factory_call_factory(self):
        SomeDerivedParsableFactory.register_widget(Widget1)
        SomeDerivedParsableFactory.register_widget(Widget2)
        try:
            SomeDerivedParsableFactory.factory()
            self.assertFalse(True)
        except ValueError as error:
            error = str(error)
            self.assertTrue(error.find('expects mandatory argument some_derived_parsable_factory_type') >= 0)
        self.assertFalse(isinstance(SomeDerivedParsableFactory.factory(some_derived_parsable_factory_type='widget1'),
                                    Widget2))
        self.assertTrue(isinstance(SomeDerivedParsableFactory.factory(some_derived_parsable_factory_type='widget1'),
                                   Widget1))
        self.assertTrue(isinstance(SomeDerivedParsableFactory.factory(some_derived_parsable_factory_type='widget1'),
                                   SomeDerivedParsableFactory))

    def test_print_help_widget_1(self):
        SomeDerivedParsableFactory.register_widget(Widget1)
        SomeDerivedParsableFactory.register_widget(Widget2)
        Widget1.print_help()
        self.assertEqual(Widget1.default_values()[Widget1.some_int],
                         413)
        self.assertEqual(Widget1.default_values()[Widget1.some_float],
                         4.13)

    def test_derived_parsable_and_base_class(self):
        SomeDerivedParsable.print_help()
        SomeParsable.print_help()

    def test_derived_parsable_and_base_class_reverse_order(self):
        SomeParsable.print_help()
        SomeDerivedParsable.print_help()

########################################################################################################################

