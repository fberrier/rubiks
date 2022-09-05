##########################################################################################
def deep_q_learning(... same as DRL ...):
    """ pseudo code for learning a puzzle via deep q learning """
    # ... same as DRL ...
    nb_actions = puzzle_type.get_nb_actions()
    while epoch < max_epochs and other_convergence_criteria(network):
        # ... same as DRL ...
        Q_and_V = dict()
        """ Compute their updated cost via value iteration ... 
             all moves are assumed to have cost 1 """
        for puzzle in puzzles:
            value = target_network(puzzle)
            actions = [0] * nb_actions
            best_action_id = 0
            if puzzle.is_goal():
                value = 0
            else:
                for action_id, child in puzzle.children():
                    child_value = target_network(child)[-1]
                    if child_value < value:
                        value = min(value, child_value)
                        best_action_id = action_id
            # We update both the value function and the best action
            actions[best_action_id] = 1
            Q_and_V[puzzle] = actions + [value]
        """ Train left-hand-side network to approximate right-hand-side target network better
        i.e. perform a forward / backward-propagation update of the network
        """
        # ... same as before ...
##########################################################################################

