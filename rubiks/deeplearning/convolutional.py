########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch import stack, unsqueeze, cat, tensor
from torch.nn import Conv2d, Sequential, ReLU, Linear, Flatten
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.utils.utils import snake_case
########################################################################################################################


class Convolutional(DeepLearning):
    """ A convolutional network """

    def get_model_details(self):
        name = [snake_case(self.__class__.__name__)]
        layers = ['%d' % l for l in self.convo_layers_description +
                  self.parallel_fully_connected_layers_description +
                  self.fully_connected_layers_description]
        ohe = ['ohe'] if self.one_hot_encoding else []
        return '_'.join(name + layers + ohe)

    one_hot_encoding = 'one_hot_encoding'
    stride = 'stride'
    padding = 'padding'
    convolution_dimension = 'convolution_dimension'
    kernel_size = 'kernel_size'
    convo_layers_description = 'convo_layers_description'
    parallel_fully_connected_layers_description = 'parallel_fully_connected_layers_description'
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
        cls.add_argument(parser,
                         field=cls.parallel_fully_connected_layers_description,
                         type=int,
                         nargs='+',
                         default=list())

    def __init__(self, **kw_args):
        DeepLearning.__init__(self, **kw_args)
        kernel_size = tuple(self.kernel_size)
        self.convo_layers_description = tuple(self.convo_layers_description)
        self.parallel_fully_connected_layers_description = tuple(self.parallel_fully_connected_layers_description)
        convo_modules = list()
        convo_layers_description = (prod(self.get_puzzle_dimension()) if self.one_hot_encoding else 1,
                                    *self.convo_layers_description)
        for in_channels, out_channels in zip(convo_layers_description[:-1],
                                             convo_layers_description[1:]):
            convo_modules.append(Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=self.padding,
                                        stride=self.stride))
            convo_modules.append(ReLU())
        self.convo_layers = Sequential(*convo_modules)
        self.layers_str = 'CL[%s]' % str(convo_layers_description)
        in_channels = prod(self.get_puzzle_dimension())
        parallel_fully_connected_layers_description = (in_channels * in_channels
                                                       if self.one_hot_encoding else in_channels,
                                                       *self.parallel_fully_connected_layers_description,
                                                       )
        self.parallel_layers = None
        if len(parallel_fully_connected_layers_description) > 1:
            parallel_layers = list()
            for x, y in zip(parallel_fully_connected_layers_description[:-1],
                            parallel_fully_connected_layers_description[1:]):
                parallel_layers.append(Linear(x, y))
                parallel_layers.append(ReLU())
            self.parallel_layers = Sequential(*parallel_layers)
            self.layers_str += '//FCL[%s]' % str(parallel_fully_connected_layers_description)
        fc_modules = list()
        if self.fully_connected_layers_description:
            fully_connected_layers_description = (*tuple(self.fully_connected_layers_description), 1)
            fc_modules.append(Flatten())
            for x, y in zip(fully_connected_layers_description[:-1],
                            fully_connected_layers_description[1:]):
                fc_modules.append(Linear(x, y))
                fc_modules.append(ReLU())
            self.layers_str += '->FCL[%s]' % str(fully_connected_layers_description)
            fc_modules = fc_modules[:-1]
        self.fc_layers = Sequential(*fc_modules)

    def get_name(self):
        name = self.__class__.__name__
        name += '[%s]' % self.layers_str
        if self.one_hot_encoding:
            name += '[one_hot_encoding]'
        return name

    def massage_puzzles(self, puzzles):
        """ We need (n_samples, in_channels=1, n, m) """
        one_hot_encoding = self.one_hot_encoding
        if isinstance(puzzles, Puzzle):
            puzzles = puzzles.to_tensor(one_hot_encoding=one_hot_encoding,
                                        flatten=False).float()
            if one_hot_encoding:
                puzzles = puzzles.permute(2, 0, 1)
            else:
                puzzles = unsqueeze(puzzles, dim=0)
            puzzles = unsqueeze(puzzles, dim=0)
        elif isinstance(puzzles, list):
            puzzles = stack([puzzle.to_tensor(one_hot_encoding=one_hot_encoding,
                                              flatten=False).float() for puzzle in puzzles])
            if one_hot_encoding:
                puzzles = puzzles.permute(0, 3, 1, 2)
            else:
                puzzles = unsqueeze(puzzles, dim=1)
        else:
            raise NotImplementedError
        return puzzles

    def forward(self, x):
        y1 = Flatten(start_dim=1)(self.convo_layers(x))
        y2 = self.parallel_layers(Flatten(start_dim=1, end_dim=3)(x)) \
            if self.parallel_layers is not None else tensor([])
        return self.fc_layers(cat((y1, y2), dim=1)).squeeze()
    
########################################################################################################################
