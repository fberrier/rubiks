########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from unittest import TestCase
########################################################################################################################


class TestArgumentParser(TestCase):

    def test_parse_int(self):
        my_int = 314
        command_line_args = "-my_int=%d" % my_int
        argv = []
        argv.extend(command_line_args.split(' '))
        parser = ArgumentParser()
        parser.add_argument('-my_int', type=int, default=None)
        parser = parser.parse_args(argv)
        self.assertEqual(my_int, parser.my_int)

    def test_parse_bool_true(self):
        command_line_args = "--my_bool"
        argv = []
        argv.extend(command_line_args.split(' '))
        parser = ArgumentParser()
        parser.add_argument("--my_bool", default=False, action='store_true')
        parser = parser.parse_args(argv)
        self.assertTrue(parser.my_bool)

    def test_parse_bool_false(self):
        argv = []
        parser = ArgumentParser()
        parser.add_argument("--my_bool", default=False, action='store_true')
        parser = parser.parse_args(argv)
        self.assertFalse(parser.my_bool)

########################################################################################################################
