##########################################################################################
def A*(initial_node, time_out, heuristic):
    """ pseudo code for A*
    g = cost_from initial_node to a given node
    h = heuristic (expected remaining cost from node to goal)
    """
    initialize_time_out(time_out)
    node = initial_node
    cost = heuristic(node) # g = 0
    frontier = sorted_multi_container({cost: {node}})
    explored = set()
    while True:
        check_time_out()
        if frontier.empty():
            return None # -> Failure to find a solution
        (cost, nodes) = frontier.pop() # pop smallest cost g+h first
        node = nodes.pop()
        if node.is_goal():
            return node
        explored.add(node)
        for child in node.children(): # -> children have g = node.g + 1
            child_cost = child.g + heuristic(child)
            if child in frontier:
                if child_cost < cost:
                    frontier.update_cost(child, child_cost)
            elif child not in explored:
                frontier.add(child, child_cost)
##########################################################################################

