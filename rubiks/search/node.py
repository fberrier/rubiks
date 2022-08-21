########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################


class Node:
    """ Node is the data structure that we manipulate in the search algorithms.
    As long as:
    - state and parent are either None or of a type that satisfies 
      the interface of BaseState
    - action is of a type that satisfies the interface of BaseAction
      and parent.action(action) == state
    Then we can run the search algorithms from this module on Nodes
    """

    def __init__(self,
                 state,
                 parent,
                 action,
                 path_cost: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        return 'State=%s, parent=%s, action=%s, path_cost=%s' % (self.state,
                                                                 self.parent,
                                                                 self.action,
                                                                 self.path_cost)

    def __hash__(self):
        """ Useful to put in containers that rely on hashes """
        return hash(self.state)

########################################################################################################################

