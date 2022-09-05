##########################################################################################
def blind_search(initial_node, time_out):
    initialize_time_out(time_out)
    """ pseudo code for BFS/DFS """
    if initial_node.is_goal():
        return initial_node
    explored = set()
    frontier = [initial_node]
    while True:
        check_time_out() # -> raise TimeoutError if appropriate
        if frontier.empty():
            """ Failure to find a solution """
            return None
        node = pop_from_frontier(frontier) # pop_from_frontier is FIFO/LIFO for BFS/DFS
        explored.add(node)
        for chile in node.children():
            if child in frontier.union(explored):
                continue
            if child.state.is_goal():
                return child
            frontier.add(child)
##########################################################################################

