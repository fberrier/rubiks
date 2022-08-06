####################################################################
from math import inf
from rubiks.puzzle.puzzle import Puzzle
####################################################################
puzzle_type=Puzzle.rubiks_cube
n=2
nb_moves=inf
print(Puzzle.factory(**globals()).apply_random_moves(nb_moves))
####################################################################

