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
    """ A fully connected network """

    one_hot_encoding = 'one_hot_encoding'
    layers_description = 'layers_description'
    default_layers = (1000, 500, 100)

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.one_hot_encoding,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.layers_description,
                         type=int,
                         nargs='+',
                         default=cls.default_layers)

    def __init__(self, **kw_args):
        DeepLearning.__init__(self, **kw_args)
        in_channels = prod(self.get_puzzle_dimension())
        if self.one_hot_encoding:
            in_channels *= in_channels
        layers = tuple(layer for layer in self.layers_description)
        if layers[-1] != 1:
            layers = (*tuple(layers), 1)
        if layers[0] != in_channels:
            layers = (in_channels, *tuple(layers))
        self.layers_int = layers
        self.layers_str = str(layers)
        modules = []
        for x, y in zip(layers[:-1], layers[1:]):
            modules.append(Linear(x, y))
            modules.append(ReLU())
        modules = modules[:-1]
        self.layers = Sequential(*modules)

    def get_name(self):
        name = self.__class__.__name__
        name += '[%s]' % self.layers_str
        if self.one_hot_encoding:
            name += '[one_hot_encoding]'
        return name

    def massage_puzzles(self, puzzles):
        if isinstance(puzzles, Puzzle):
            puzzles = puzzles.to_tensor(one_hot_encoding=self.one_hot_encoding).float().\
                reshape(-1).unsqueeze(0)
        elif isinstance(puzzles, list):
            puzzles = stack([puzzle.to_tensor(one_hot_encoding=self.one_hot_encoding).float().
                            reshape(-1) for puzzle in puzzles])
        else:
            raise NotImplementedError
        return puzzles

    def forward(self, x):
        return self.layers(x).squeeze()
    
########################################################################################################################
