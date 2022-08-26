from math import inf
from rubiks.puzzle.rubikscube import RubiksCube
from rubiks.solvers.kociembasolver import KociembaSolver
cubes = RubiksCube(n=3).apply_random_moves(inf).get_equivalent()
solver = KociembaSolver(n=3)
ok = 0
for cube in cubes:
    solution = solver.solve(cube)
    if solution.success:
        ok += 1
    print(solution)
print(ok)