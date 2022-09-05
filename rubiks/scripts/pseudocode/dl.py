##########################################################################################
def deep_learning(puzzle_type,
                  teacher_heuristic,
                  loss_function,
                  network_architecture):
    """ pseudo code for learning a puzzle via deep learning and a teacher heuristic """
    target_value_function = {state: teacher_heuristic(state)
                             for state in generate_random_states(puzzle_type)}
    dl_heuristic = train_neural_network(target_value_function,
                                        loss_function,
                                        network_architecture)
    return dl_heuristic
##########################################################################################

