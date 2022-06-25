########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.core.parsable import Parsable
from rubiks.puzzle.puzzle import Puzzle
########################################################################################################################


class Puzzled(Parsable):

    puzzle_type = Puzzle.puzzle_type

    @classmethod
    def populate_parser_impl(cls, parser):
        Puzzle.populate_parser(parser)

    @classmethod
    def additional_dependencies(cls):
        return Puzzle.get_widgets()

    def __init__(self, **kw_args):
        Parsable.__init__(self, **kw_args)
        self.__goal__ = Puzzle.factory(**kw_args)
        self.pp_nb = self.__goal__.possible_puzzles_nb()

    def get_puzzle_type(self):
        return self.puzzle_type

    def get_puzzle_type_class(self):
        return type(self.__goal__)

    def get_puzzle_dimension(self):
        return self.__goal__.dimension()

    def get_goal(self):
        return self.__goal__.clone()

    def puzzle_name(self):
        return self.__goal__.get_name()

    def possible_puzzles_nb(self):
        return self.pp_nb

########################################################################################################################
