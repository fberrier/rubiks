########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch import stack, cat
from torch.nn import Sequential, ReLU, BatchNorm1d, Linear, Dropout
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.utils.utils import snake_case
########################################################################################################################


class FullyConnected(DeepLearning):
    """ A fully connected network """

    def get_model_details(self):
        name = [snake_case(self.__class__.__name__)]
        layers = ['%d' % l for l in self.layers_description]
        ohe = ['ohe'] if self.one_hot_encoding else []
        drop_out = ['do%d' % int(100*self.drop_out)] if 0.0 < self.drop_out < 1.0 else []
        return '_'.join(name + layers + ohe + drop_out)

    one_hot_encoding = 'one_hot_encoding'
    layers_description = 'layers_description'
    default_layers = (1000, 500, 100)
    drop_out = 'drop_out'

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
        cls.add_argument(parser,
                         field=cls.drop_out,
                         type=float,
                         default=0.0)

    def __init__(self, **kw_args):
        DeepLearning.__init__(self, **kw_args)
        in_channels = self.number_of_tiles()
        if self.one_hot_encoding:
            in_channels *= self.number_of_values()
        layers = tuple(layer for layer in self.layers_description)
        if layers[-1] != 1:
            layers = (*tuple(layers), 1)
        if layers[0] != in_channels:
            layers = (in_channels, *tuple(layers))
        self.layers_str = str(layers)
        modules = list()
        for x, y in zip(layers[:-1], layers[1:]):
            modules.append(Linear(x, y))
            modules.append(ReLU())
            if self.drop_out and 0 < self.drop_out < 1.0:
                modules.append(Dropout(self.drop_out))
        modules = modules[:-2 if self.drop_out else -1]
        self.policy = None
        self.value_function = None
        if self.joint_policy:
            cut = -3 if not self.drop_out else -4
            self.layers = Sequential(*modules[:cut])
            self.value_function = Sequential(*modules[cut:])
            self.policy = Sequential(*modules[cut:-1], Linear(layers[-2], self.nb_moves()))
        else:
            self.layers = Sequential(*modules)

    def get_name(self):
        name = self.__class__.__name__
        name += '[%s]' % self.layers_str
        if self.one_hot_encoding:
            name += '[one_hot_encoding]'
        if 0.0 < self.drop_out < 1.0:
            name += '[drop_out=%s]' % self.drop_out
        if self.joint_policy:
            name += '[jp&v]'
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
        y = self.layers(x)
        if not self.joint_policy:
            return y.squeeze()
        policy = self.policy(y)
        value = self.value_function(y)
        return cat((value, policy), dim=1).squeeze()
    
########################################################################################################################
