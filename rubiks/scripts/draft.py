####################################################################
from rubiks.puzzle.rubikscube import RubiksCube
####################################################################
cube = RubiksCube(n=3)
print(cube)
print(cube.__hash__())
print(hash(cube))
print(cube.goals_hashes)
print(cube.is_goal())