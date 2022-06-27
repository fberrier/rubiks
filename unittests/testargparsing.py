########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from unittest import TestCase
########################################################################################################################


class TestArgumentParser(TestCase):

    def test_parse_silly_example(self):
        parser = ArgumentParser()
        parser.add_argument('-a', type=int, default=None)
        parser.add_argument('-aa', type=int, default=None)
        command_line_args = "-a=32 -aa=34"
        argv = []
        argv.extend(command_line_args.split(' '))
        self.assertEqual(32, parser.parse_args(argv).a)
        self.assertEqual(34, parser.parse_args(argv).aa)

    def test_parse_silly_example_2(self):
        parser = ArgumentParser()
        parser.add_argument('-aa', type=int, default=None)
        parser.add_argument('-a', type=int, default=None)
        command_line_args = "-a=32 -aa=34"
        argv = []
        argv.extend(command_line_args.split(' '))
        self.assertEqual(32, parser.parse_args(argv).a)
        self.assertEqual(34, parser.parse_args(argv).aa)

    def test_parse_int(self):
        parser = ArgumentParser()
        parser.add_argument('-my_int', type=int, default=None)
        my_int = 314
        command_line_args = "-my_int=%d" % my_int
        argv = []
        argv.extend(command_line_args.split(' '))
        self.assertEqual(my_int, parser.parse_args(argv).my_int)
        my_int2 = 315
        command_line_args = "-my_int=%d" % my_int2
        argv = []
        argv.extend(command_line_args.split(' '))
        self.assertEqual(my_int2, parser.parse_args(argv).my_int)

    def test_parse_int_float_repeat(self):
        parser = ArgumentParser()
        parser.add_argument('-my_int', type=int, default=None)
        parser.add_argument('-my_float', type=float, default=1.0)
        my_int = 314
        my_float = 3.0
        command_line_args = "-my_int=%d" % my_int
        command_line_args += " -my_float=%d" % my_float
        argv = []
        argv.extend(command_line_args.split(' '))
        first_time = parser.parse_args(argv)
        self.assertEqual(my_int, first_time.my_int)
        self.assertEqual(my_float, first_time.my_float)
        my_int2 = 315
        command_line_args = "-my_int=%d" % my_int2
        argv = []
        argv.extend(command_line_args.split(' '))
        second_time = parser.parse_args(argv)
        self.assertEqual(my_int2, second_time.my_int)
        self.assertEqual(1.0, second_time.my_float)

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
