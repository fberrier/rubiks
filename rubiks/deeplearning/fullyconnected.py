########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch import stack
from torch.nn import Sequential, ReLU, BatchNorm1d, Linear
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class FullyConnected(DeepLearning):
    """ TBD """

    network_type = DeepLearning.fully_connected_net

    def __init__(self, puzzle_type, **kw_args):
        DeepLearning.__init__(self, puzzle_type, **kw_args)
        in_channels = prod(self.puzzle_dimension) ** 2
        layers = kw_args.get('layers', (1000, 500, 100))
        if layers[-1] != 1:
            layers = (*tuple(layers), 1)
        if layers[0] != in_channels:
            layers = (in_channels, *tuple(layers))
        self.layers_str = str(layers)
        modules = []
        for x, y in zip(layers[:-1], layers[1:]):
            modules.append(Linear(x, y))
            modules.append(ReLU())
        modules = modules[:-1]
        self.layers = Sequential(*modules)

    def name(self):
        name = self.__class__.__name__
        if hasattr(self, 'layers'):
            name += '[%s]' % self.layers_str
        return name

    def massage_puzzles(self, puzzles):
        if isinstance(puzzles, Puzzle):
            puzzles = puzzles.to_tensor().float().reshape(-1).unsqueeze(0)
        elif isinstance(puzzles, list):
            puzzles = stack([puzzle.to_tensor().float().reshape(-1) for puzzle in puzzles])
        else:
            raise NotImplementedError
        return puzzles

    def forward(self, x):
        return self.layers(x).squeeze()
    
########################################################################################################################
