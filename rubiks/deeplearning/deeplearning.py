########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from copy import deepcopy as copy
from pandas import read_pickle
from torch import device
from torch.nn import Module
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.core.factory import Factory
from rubiks.puzzle.puzzled import Puzzled
from rubiks.utils.utils import to_pickle
########################################################################################################################


class DeepLearning(Module, Factory, Puzzled, Loggable, metaclass=ABCMeta):
    """ TBD """

    use_cuda = 'use_cuda'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.use_cuda,
                         default=False,
                         action=cls.store_true)

    def __init__(self, **kw_args):
        Module.__init__(self)
        Puzzled.__init__(self, **kw_args)
        Loggable.__init__(self, **kw_args)
        self.cuda_device = None

    network_type = 'network_type'
    fully_connected_net = 'fully_connected_net'
    state_dict_tag = 'state_dict'

    @classmethod
    def factory_key_name(cls):
        return cls.network_type

    def set_cuda(self):
        if self.use_cuda and not self.cuda_device:
            self.cuda()
            self.cuda_device = device('cuda:0')

    @classmethod
    def widget_types(cls):
        from rubiks.deeplearning.fullyconnected import FullyConnected
        return {cls.fully_connected_net: FullyConnected}

    @classmethod
    def factory_impl(cls, network_type, **kw_args):
        if cls.fully_connected_net == network_type:
            from rubiks.deeplearning.fullyconnected import FullyConnected as Network
        else:
            raise NotImplementedError('DeepLearning.factory cannot construct network of type %s' % network_type)
        return Network(**kw_args)

    def save(self, model_file_name):
        cls = self.__class__
        data = {cls.puzzle_type: self.puzzle_type,
                cls.network_type: self.network_type,
                cls.kw_args: self.kw_args,
                cls.state_dict_tag: self.state_dict()}
        to_pickle(data, model_file_name)

    @classmethod
    def restore(cls, model_file):
        data = read_pickle(model_file)
        deeplearning = cls.factory(data[Puzzled.puzzle_type],
                                   data[cls.network_type],
                                   **data[cls.kw_args])
        deeplearning.load_state_dict(data[cls.state_dict_tag])
        return deeplearning

    def get_name(self):
        return self.__class__.__name__

    def evaluate(self, puzzles):
        self.set_cuda()
        puzzles = self.massage_puzzles(puzzles)
        if self.use_cuda:
            puzzles = puzzles.to(self.cuda_device)
        return self.forward(puzzles)

    @abstractmethod
    def massage_puzzles(self, puzzle):
        """ need to specify in concrete implementations. Can e.g. get tensor from puzzle and manipulate it, etc...
        """
        return

    def clone(self):
        cloned = self.__class__(**self.get_config())
        cloned.load_state_dict(copy(self.state_dict()))
        return cloned
    
########################################################################################################################
