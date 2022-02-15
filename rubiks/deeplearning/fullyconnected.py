########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch.nn import Sequential, ReLU, BatchNorm1d, Linear
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class FullyConnected(DeepLearning):
    """ TBD """

    def __init__(self, puzzle_dimension, **kw_args):
        DeepLearning.__init__(self, puzzle_dimension, **kw_args)
        in_channels = prod(self.puzzle_dimension) ** 2
        self.layers = Linear(16, 1)
        #Sequential(Linear(in_channels, 1000),
         #                        ReLU(),
          #                       BatchNorm1d(1000),
           #                      Linear(1000, 500),
            #                     ReLU(),
             #                    BatchNorm1d(500),
              #                   Linear(500, 100),
               #                  ReLU(),
                #                 BatchNorm1d(100),
                 #                Linear(100, 1))

    def evaluate(self, puzzle):
        puzzle = puzzle.to_tensor().float().reshape(-1)
        self.log_info(puzzle)
        value = self.layers(puzzle)
        return value

    def forward(self, x):
        return self.layers(x)
    
########################################################################################################################
