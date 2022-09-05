##########################################################################################
def deep_reinforcement_learning(puzzle_type,
                                loss_function,
                                max_epochs,
                                target_network_update_criteria,
                                puzzles_generation_criteria,
                                network_architecture):
    """ pseudo code for learning a puzzle via deep reinforcement learning """
    network = get_network(puzzle_type, network_architecture)
    target_network = get_network(puzzle_type, network_architecture)
    epoch = 0
    puzzles = list()
    while epoch < max_epochs and other_convergence_criteria(network):
        epoch += 1
        """ Generate a new bunch of puzzles """
        if puzzles_generation_criteria(puzzles):
            puzzles = generate_random_states(puzzle_type)
        V = dict()
        """ Compute their updated cost via value iteration ... 
             all moves are assumed to have cost 1 """
        for puzzle in puzzles:
            if puzzle.is_goal():
                V[puzzle] = 0
            else:
                V[puzzle] = min(target_network(puzzle),
                                1 + min(target_network(child) for child in puzzle.children()))
        """ Train left-hand-side network to approximate right-hand-side target network better
        i.e. perform a forward / backward-propagation update of the network
        """
        network = train_neural_network(V,
                                       loss_function,
                                       network_architecture)
        """ Update the target network if criteria are met 
        (e.g. epochs, convergence, no progress, etc...) """
        if target_network_update_criteria(network, target_network):
            target_network = copy(network)
    return network
##########################################################################################

