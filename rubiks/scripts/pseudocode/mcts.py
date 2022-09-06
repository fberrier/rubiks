#################################################################
def MCTS(initial_node, time_out, q_v_network):
    """ pseudo code for Monte Carlo Tree Search """
    initialize_time_out(time_out)
    tree = Tree(initial_node)
    while True:
        check_time_out()
        leaf = construct_path_to_new_leaf(tree, q_v_network)
        if leaf.is_goal():
            return BFS(tree) # path to leaf can usually be improved by BFS
#################################################################
def construct_path_to_new_leaf(tree, q_v_network):
    leaf = tree.initial_node
    while leaf.expanded:
        actions_proba, value = q_v_network(leaf)
        historical_actions_count = leaf.actions_taken
        leaf = leaf.choose_child(actions_proba,
                                 value,
                                 historical_actions_count)
    # add children of leaf to tree
    leaf.expand()
    # penalize path taken to favour exploration
    leaf.increment_historical_actions_count()
    return leaf
#################################################################