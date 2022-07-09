########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch import stack, unsqueeze
from torch.nn import Conv3d, Conv2d, Sequential, ReLU, Linear, Flatten, Dropout
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.utils.utils import snake_case
########################################################################################################################


class Convolutional(DeepLearning):
    """ A convolutional network """

    def get_model_details(self):
        name = [snake_case(self.__class__.__name__)]
        ohe = ['ohe'] if self.one_hot_encoding else []
        return '_'.join(name + ohe)

    one_hot_encoding = 'one_hot_encoding'
    stride = 'stride'
    padding = 'padding'
    convolution_dimension = 'convolution_dimension'
    kernel_size = 'kernel_size'
    convo_layers_description = 'convo_layers_description'
    fully_connected_layers_description = 'fully_connected_layers_description'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.one_hot_encoding,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.convolution_dimension,
                         type=int,
                         default=2)
        cls.add_argument(parser,
                         field=cls.padding,
                         type=int,
                         default=0)
        cls.add_argument(parser,
                         field=cls.stride,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         field=cls.kernel_size,
                         type=int,
                         nargs='+',
                         default=2)
        cls.add_argument(parser,
                         field=cls.convo_layers_description,
                         type=int,
                         nargs='+')
        cls.add_argument(parser,
                         field=cls.fully_connected_layers_description,
                         type=int,
                         nargs='+',
                         default=list())

    def __init__(self, **kw_args):
        DeepLearning.__init__(self, **kw_args)
        self.kernel_size = tuple(self.kernel_size)
        self.convo_layers_description = tuple(self.convo_layers_description)
        if self.one_hot_encoding:
            self.convolution_dimension = 3
            convo_layer = Conv3d
        else:
            self.convolution_dimension = 2
            convo_layer = Conv2d
        modules = list()
        self.convo_layers_description = (1, *self.convo_layers_description)
        for in_channels, out_channels in zip(self.convo_layers_description[:-1],
                                             self.convo_layers_description[1:]):
            modules.append(convo_layer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.padding,
                                       stride=self.stride))
            modules.append(ReLU())
        if self.fully_connected_layers_description:
            self.fully_connected_layers_description = (*tuple(self.fully_connected_layers_description), 1)
            modules.append(Flatten())
        for x, y in zip(self.fully_connected_layers_description[:-1],
                        self.fully_connected_layers_description[1:]):
            modules.append(Linear(x, y))
            modules.append(ReLU())
        self.layers = Sequential(*modules)

    def get_name(self):
        name = self.__class__.__name__
        name += '[%s]' % self.layers_str
        if self.one_hot_encoding:
            name += '[one_hot_encoding]'
        return name

    def massage_puzzles(self, puzzles):
        """ We need (n_samples, in_channels=1, n, m) """
        if isinstance(puzzles, Puzzle):
            puzzles = puzzles.to_tensor(one_hot_encoding=self.one_hot_encoding,
                                        flatten=False).float()
            puzzles = unsqueeze(puzzles, dim=0)
            puzzles = unsqueeze(puzzles, dim=0)
        elif isinstance(puzzles, list):
            puzzles = stack([puzzle.to_tensor(one_hot_encoding=self.one_hot_encoding,
                                              flatten=False).float() for puzzle in puzzles])
            puzzles = unsqueeze(puzzles, dim=1)
        else:
            raise NotImplementedError
        return puzzles

    def forward(self, x):
        return self.layers(x).squeeze()
    
########################################################################################################################
