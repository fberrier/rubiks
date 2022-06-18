########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from copy import deepcopy as copy
from pandas import to_pickle, read_pickle
from torch import device
from torch.nn import Module
########################################################################################################################
from rubiks.utils.loggable import Loggable
from rubiks.utils.utils import touch
########################################################################################################################


class DeepLearning(Module, Loggable, metaclass=ABCMeta):
    """ TBD """

    network_type = None

    def __init__(self, puzzle_type, **kw_args):
        Loggable.__init__(self, self.name())
        self.puzzle_type = puzzle_type
        self.kw_args = kw_args
        self.puzzle_dimension = self.puzzle_type.construct_puzzle(**kw_args).dimension()
        Module.__init__(self)
        assert self.network_type is not None, 'Concrete DeepLearning should have a proper network_type'
        self.use_cuda = self.kw_args.get('use_cuda', False)
        self.cuda_device = None

    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    network_type_tag = 'network_type_tag'
    kw_args = 'kw_args'
    fully_connected_net = 'fully_connected_net'
    state_dict_tag = 'state_dict_tag'

    def set_cuda(self):
        if self.use_cuda and not self.cuda_device:
            self.cuda()
            self.cuda_device = device('cuda:0')
        
    @classmethod
    def factory(cls, puzzle_type, network_type, **kw_args):
        if cls.fully_connected_net == network_type:
            from rubiks.deeplearning.fullyconnected import FullyConnected
            network = FullyConnected
        else:
            raise NotImplementedError('DeepLearning.factory cannot construct network of type %s' % network_type)
        return network(puzzle_type, **kw_args)

    def save(self, model_file_name):
        cls = self.__class__
        data = {cls.puzzle_type: self.puzzle_type,
                cls.network_type_tag: self.network_type,
                cls.kw_args: self.kw_args,
                cls.state_dict_tag: self.state_dict()}
        touch(model_file_name)
        to_pickle(data, model_file_name)

    @classmethod
    def restore(cls, model_file):
        data = read_pickle(model_file)
        deeplearning = cls.factory(data[cls.puzzle_type],
                                   data[cls.network_type_tag],
                                   **data[cls.kw_args])
        deeplearning.load_state_dict(data[cls.state_dict_tag])
        return deeplearning

    def name(self):
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
        cloned = self.__class__(self.puzzle_type, **self.kw_args)
        cloned.load_state_dict(copy(self.state_dict()))
        return cloned
    
########################################################################################################################
