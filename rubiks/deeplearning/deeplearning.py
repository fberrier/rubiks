########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from numpy import isnan
from pandas import DataFrame
from time import time as snap
########################################################################################################################
from rubiks.utils.loggable import Loggable
########################################################################################################################


class DeepLearning(Loggable, metaclass=ABCMeta):
    """ TBD """

    def __init__(self, **kw_args):
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))
        self.network = None

    network_config = 'network_config'
    network_type = 'network_type'
    fully_connected_net = 'fully_connected_net'
        
    @classmethod
    def factory(cls, **kw_args):
        if cls.network_config not in kw_args:
            raise ValueError('DeepLearning.factory cannot construct network as there is no %s' % cls.network_config)
        network_config = kw_args[cls.network_config]
        if not isinstance(network_config, dict) or cls.network_type not in network_config:
            raise ValueError('DeepLearning.factory cannot construct network as %s badly formed' % cls.network_config)
        network_type = network_config[cls.network_type]
        if cls.fully_connected_net == network_type:
            from rubiks.deeplearning.fullyconnected import FullyConnected
            network = FullyConnected
        else:
            raise NotImplementedError('DeepLearning.factory cannot construct network of type %s' % network_type)
        return network(**network_config)

    def save(self, data_base):
        """ overwrite where meaningful """
        return

    @staticmethod
    def restore(data_base):
        """ overwrite where meaningful """
        return

    def name(self):
        return self.__class__.__name__

    def evaluate(self, puzzle):
        return self.network(puzzle.to_tensor())
    
 ########################################################################################################################
