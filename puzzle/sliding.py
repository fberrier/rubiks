########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from random import randint
import numpy as np
########################################################################################################################
from puzzle.puzzle import Move, Puzzle
########################################################################################################################


class Slide(Move):

    def __init__(self, n, m):
        self.tile = (n, m)

########################################################################################################################

    
class SlidingPuzzle(Puzzle):
    """ Game of the sliding Puzzle, e.g. the 8-puzzle, 15-puzzle, etc """

    move_type = Slide

    possible_moves = {}

    def __init__(self, tiles, empty=None):
        super().__init__()
        self.tiles = tiles
        if empty is None:
            self.empty = tuple(np.argwhere(0 == tiles)[0])
        else:
            self.empty = empty

    def __repr__(self):
        return 'tiles=%s\nempty=%s' % (self.tiles, self.empty)

    @staticmethod
    def goal(n, m=None):
        if m is None:
            m = n
        goal = np.array(range(1, n * m + 1)).reshape((n, m))
        goal[n - 1][m - 1] = 0
        return SlidingPuzzle(goal, (n - 1, m - 1))
            
    def goal_state(self):
        return self.goal(*self.tiles.shape)

    def apply(self, move: Slide):
        """ moved tile must either be same row or same col as the empty tile 
        and next to it. If they are, we swap empty with slide and return
        """
        mt0 = move.tile[0]
        mt1 = move.tile[1]
        if any(mt < 0 for mt in [mt0, mt1]) or mt0 >= self.tiles.shape[0] or mt1 >= self.tiles.shape[1]:
            raise ValueError('Invalid slide')
        delta_n = mt0 - self.empty[0]
        delta_m = mt1 - self.empty[1]
        if 0 == delta_n:
            if delta_m not in {1, -1}:
                raise ValueError('Invalid slide')
        elif 0 == delta_m:
            if delta_n not in {1, -1}:
                raise ValueError('Invalid slide')
        else:  
            raise ValueError('Invalid slide')
        tiles = self.tiles.copy()
        tiles[self.empty[0]][self.empty[1]] = self.tiles[move.tile[0]][move.tile[1]]
        empty = move.tile
        tiles[empty[0]][empty[1]] = 0
        return SlidingPuzzle(tiles, empty)

    @staticmethod
    def choices(empty, shape):
        c = []
        if empty[0] > 0:
            c.append(Slide(empty[0] - 1, empty[1]))
        if empty[0] < shape[0] - 1:
            c.append(Slide(empty[0] + 1, empty[1]))
        if empty[1] > 0:
            c.append(Slide(empty[0], empty[1] - 1))
        if empty[1] < shape[1] - 1:
            c.append(Slide(empty[0], empty[1] + 1))
        return c

    def random_move(self):
        if self.tiles.shape not in self.possible_moves:
            self.possible_moves[self.tiles.shape] = self.get_possible_moves(self.tiles.shape)
        choices = self.possible_moves[self.tiles.shape][self.empty]
        return choices[randint(0, len(choices) - 1)]

    def get_possible_moves(self, shape):
        possible_moves = {}
        for row in range(shape[0]):
            for col in range(shape[1]):
                empty = (row, col)
                possible_moves[empty] = self.choices(empty, shape)
        return possible_moves
            
########################################################################################################################
