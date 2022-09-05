# ###################################################################
from math import inf
# ###################################################################
from rubiks . deeplearning . deeplearning import DeepLearning
from rubiks . learners . learner import Learner
from rubiks . learners . deeplearner import DeepLearner
from rubiks . puzzle . puzzle import Puzzle
from rubiks . puzzle . trainingdata import TrainingData
# ###################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle . rubiks_cube
    n=2
    init_from_random_goal = False
    goals = Puzzle.factory(**globals()).get_equivalent()
    print(goals[0], goals[-1])
# ##################################################################