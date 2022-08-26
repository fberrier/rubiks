########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from torch import Tensor
########################################################################################################################
from rubiks.puzzle.puzzled import Puzzled
from rubiks.core.factory import Factory
########################################################################################################################


class Heuristic(Factory, Puzzled, metaclass=ABCMeta):
    """ Generic concept of a heuristic: a class that is able to give us an estimated cost-to-go for a particular
    type of Puzzle of a given dimension.
    e.g. could be typical heuristics used in A* searches such as those based on Manhattan distance or similar concepts
    or could be neural net based heuristics that have been trained by Deep Reinforcement Learning.
    """

    heuristic_type = 'heuristic_type'
    manhattan = 'manhattan'
    deep_learning = 'deep_learning'
    deep_q_learning = 'deep_q_learning'
    perfect = 'perfect'
    kociemba = 'kociemba'
    known_heuristic_types = [manhattan, deep_learning, deep_q_learning, perfect, kociemba]

    def __init__(self, **kw_args):
        Factory.__init__(self, **kw_args)
        Puzzled.__init__(self, **kw_args)

    @classmethod
    def factory_key_name(cls):
        return cls.heuristic_type

    @classmethod
    def widget_types(cls):
        from rubiks.heuristics.perfectheuristic import PerfectHeuristic
        from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
        from rubiks.heuristics.deepqlearningheuristic import DeepQLearningHeuristic
        from rubiks.heuristics.manhattan import Manhattan
        from rubiks.heuristics.kociembaheuristic import KociembaHeuristic
        return {cls.perfect: PerfectHeuristic,
                cls.deep_learning: DeepLearningHeuristic,
                cls.deep_q_learning: DeepQLearningHeuristic,
                cls.manhattan: Manhattan,
                cls.kociemba: KociembaHeuristic}

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.heuristic_type,
                         choices=cls.known_heuristic_types,
                         default=cls.manhattan)

    known_heuristics = [manhattan, deep_learning, perfect]

    @classmethod
    @abstractmethod
    def known_to_be_admissible(cls):
        return False

    @abstractmethod
    def cost_to_go_from_puzzle_impl(self, puzzle):
        """ this is where derived concrete classes should implement the actual heuristic """
        return

    dim_error_message = 'cost_to_go expected %s of dimension %s. Got %s instead'

    def cost_to_go_from_puzzle(self, puzzle):
        assert puzzle.dimension() == self.get_puzzle_dimension(),\
            self.dim_error_message % (str(self.get_puzzle_type()),
                                      str(self.get_puzzle_dimension()),
                                      str(puzzle.dimension()))
        return self.cost_to_go_from_puzzle_impl(puzzle)

    def cost_to_go_from_tensor(self, puzzle, **kw_args):
        """ feel free to overwrite this by more efficient in derived classes. If not, we convert to 
        puzzle from tensor from the function that the puzzle (has to) provide
        """
        return self.cost_to_go_from_puzzle(self.puzzle_type.from_tensor(puzzle, **kw_args))

    def cost_to_go(self, puzzle, **kw_args):
        if isinstance(puzzle, Tensor):
            return self.cost_to_go_from_tensor(puzzle, **kw_args)
        return self.cost_to_go_from_puzzle(puzzle)

    def get_name(self):
        return self.__class__.__name__

########################################################################################################################

