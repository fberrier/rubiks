#####################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.solvers.kociembasolver import KociembaSolver
from_kociemba = KociembaSolver.from_kociemba
to_kociemba = KociembaSolver.to_kociemba
#####################################################################
puzzle_type = Puzzle.rubiks_cube
n=3
init_from_random_goal=False
cube = Puzzle.factory(**globals()).get_equivalent()[-1].apply_random_moves(nb_moves=1)
solver_type = Solver.kociemba
solver = Solver.factory(**globals())
print(solver.solve(cube))
###################################################################

