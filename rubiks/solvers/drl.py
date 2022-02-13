########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from torch import tensor
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.solvers.solver import Solver
########################################################################################################################


class DRLSolver(Solver):
    """ This solver will learn the cost-to-go via deep reinforcement learning """


    def __init__(self, puzzle_type, nb_epochs, nb_shuffles, nb_sequences, **kw_args):
        Solver.__init__(self, puzzle_type, **kw_args)
        self.nb_epochs = nb_epochs
        self.nb_shuffles = nb_shuffles
        self.nb_sequences = nb_sequences
        self.target_network = DeepLearning.factory(**kw_args)
        self.current_network = DeepLearning.factory(**kw_args)

    def learn(self):
        self.log_info('Learn')
        for epoch in range(1, self.nb_epochs + 1):
            self.log_info('epoch ', epoch)
            training_data = self.puzzle_type.get_training_data(nb_shuffles=self.nb_shuffles,
                                                               nb_sequences=self.nb_sequences,
                                                               **self.kw_args)
            self.log_info(training_data)
            (puzzles, nb_shuffles_List) = training_data
            targets = []
            hashes_to_puzzle = {hash(puzzle): (puzzle, nb_shuffles) for puzzle, nb_shuffles in zip(*training_data)}
            for puzzle, nb_shuffles in zip(*training_data):
                self.log_info('Checking out puzzle\n', puzzle, ' with nb_shuffles: ', nb_shuffles)
                if 0 == nb_shuffles or puzzle.is_goal():
                    target = 0
                elif 1 == nb_shuffles:
                    # cuz can't do better knowledge than that
                    target = 1
                else:
                    target = float('inf')
                    one_away_puzzles = [puzzle.apply(move) for move in puzzle.possible_moves()]
                    for one_away_puzzle in one_away_puzzles:
                        candidate_target = self.target_network.evaluate(one_away_puzzle)
                        one_away_puzzle_hash = hash(one_away_puzzle)
                        if one_away_puzzle_hash in hashes_to_puzzle:
                            candidate_target = min(candidate_target, hashes_to_puzzle[one_away_puzzle_hash][1])
                        candidate_target += 1
                        target = min(target, candidate_target)
                targets.append(target)
            targets = tensor(targets)
            puzzles = tensor([puzzle.to_tensor() for puzzle in puzzles])
            for puzzle, target in zip(puzzles, targets):
                self.log_info('puzzle:\n', puzzle, ', target: ', target)
                

    def solve_impl(self, puzzle, time_out, **kw_args):
        return 0


########################################################################################################################
