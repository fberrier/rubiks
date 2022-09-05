##########################################################################################
def MCTS(initial_node, time_out, q_v_network):
    """ pseudo code for Monte Carlo Tree Search """
    initialize_time_out(time_out)
    tree = Tree(initial_node)
    while True:
        check_time_out()
        leaf = add_new_leaf(tree, q_v_network)
        if leaf.is_goal():
            return BFS(tree) # path to leaf can usually be improved by BFS
##########################################################################################
def add_new_leaf(tree, q_v_network):
    leaf = tree.initial_node
    while leaf.expanded:
        pass
    leaf.expand() # add children of leaf to tree
    leaf.update_penalties()
    return leaf
##########################################################################################

