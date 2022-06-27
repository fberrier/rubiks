########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from copy import deepcopy as copy
from torch import device
from torch.nn import Module
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.core.factory import Factory
from rubiks.puzzle.puzzled import Puzzled
from rubiks.utils.utils import get_model_file_name
########################################################################################################################


class DeepLearning(Module, Factory, Puzzled, Loggable, metaclass=ABCMeta):
    """ Essentially a Factory around Pytorch to instantiate networks """

    use_cuda = 'use_cuda'

    def __init__(self, **kw_args):
        Module.__init__(self)
        Factory.__init__(self, **kw_args)
        Puzzled.__init__(self, **kw_args)
        Loggable.__init__(self, **kw_args)
        self.cuda_device = None

    network_type = 'network_type'
    fully_connected_net = 'fully_connected_net'
    state_dict_tag = 'state_dict'

    @abstractmethod
    def get_model_details(self):
        """ Return something that can identify in the name of a file which config this model was using """
        return

    def get_model_name(self):
        return get_model_file_name(self.get_puzzle_type(),
                                   self.get_puzzle_dimension(),
                                   model_name=self.get_model_details())

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.use_cuda,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.network_type,
                         choices=[cls.fully_connected_net])

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

    def get_data(self):
        return (self.get_config(),
                self.state_dict())

    @classmethod
    def restore(cls, data):
        deeplearning = cls.factory(**data[0])
        deeplearning.load_state_dict(data[1])
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
