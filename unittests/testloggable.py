########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
########################################################################################################################


class DerivedLoggable(Loggable):

    some_other_name = 'SomeOtherName'

    def get_name(self):
        return __class__.some_other_name

########################################################################################################################


class TestLoggable(TestCase):

    def test_loggable(self):
        for log_level in Loggable.known_log_levels:
            my_loggable = Loggable(log_level=log_level, name=log_level)
            my_loggable.log_debug('nothing to say really')
            my_loggable.log_info('nothing to say really')
            my_loggable.log_warning('nothing to say really')
            my_loggable.log_error('nothing to say really')
            self.assertEqual(log_level, my_loggable.log_level)

    def test_derived_loggable(self):
        derived_loggable = DerivedLoggable(log_level=Loggable.debug)
        self.assertNotEqual(DerivedLoggable.some_other_name, derived_loggable.name)
        derived_loggable.log_info('nothing much')
        self.assertEqual(DerivedLoggable.some_other_name,
                         derived_loggable.name)
        derived_loggable.log_info('how about now?')

    def test_log_dict(self):
        logger = Loggable(name='test_log_dict')
        what = {'a': 'toto', 'b': 'titi'}
        logger.log_info(what)
        what = {'a': 2, 'b': 'titi'}
        logger.log_info(what)
        what = {'a': 2, 'b': [4]}
        logger.log_info(what)
        what = {'a': 2, 'b': [4, 5, 6]}
        logger.log_info(what)
        logger.log_info({'expected_column': [1, 5, 9, 13],
                         'actual_column': [1, 5, 9, 13]})

########################################################################################################################

