####################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
####################################################################
puzzle_type = Puzzle.rubiks_cube
n = 2
nb_moves = 5
init_from_random_goal = True
solver_type = Solver.bfs
time_out = 360
cube = Puzzle.factory(**globals())
cube = cube.apply_random_moves(nb_moves=nb_moves)
solver = Solver.factory(**globals())
logger = Loggable(name='draft')
solution = solver.solve(puzzle=cube)
logger.log_info(solution)
####################################################################