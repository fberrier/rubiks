##########################################################################################
def value_iteration(states):
    """ pseudo code for RL value iteration """
    V = {state: inf for state in states}
    change = True
    while change:
        change = False
        for state, cost in V:
            V[state] = min(V[child] + t_cost for child, t_cost in state.children())
            change |= V[state] < cost
    return V
##########################################################################################

