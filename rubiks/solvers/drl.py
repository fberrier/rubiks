########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from torch import tensor, no_grad
from torch.nn import MSELoss
from torch.optim import RMSprop
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
        self.target_network = DeepLearning.factory(puzzle_type, **kw_args)
        self.current_network = DeepLearning.factory(puzzle_type, **kw_args)
        self.update_target_network_frequency = kw_args.pop('update_target_network_frequency', 100)
        self.loss_function = MSELoss()
        self.learning_rate = kw_args.pop('learning_rate', 1e-6)
        self.log_info('learning_rate: ', self.learning_rate)

    def learn(self):
        optimizer = RMSprop(self.current_network.parameters(),
                            lr=self.learning_rate)
        for epoch in range(1, self.nb_epochs + 1):
            training_data = self.puzzle_type.get_training_data(nb_shuffles=self.nb_shuffles,
                                                               nb_sequences=self.nb_sequences,
                                                               **self.kw_args)
            (puzzles, nb_shuffles_List) = training_data
            targets = []
            hashes_to_puzzle = {}
            for puzzle, nb_shuffles in zip(*training_data):
                puzzle_hash = hash(puzzle)
                if puzzle_hash not in hashes_to_puzzle:
                    hashes_to_puzzle[puzzle_hash] = (puzzle, nb_shuffles)
                elif hashes_to_puzzle[puzzle_hash][1] > nb_shuffles:
                    hashes_to_puzzle[puzzle_hash][1] = nb_shuffles
            for puzzle, nb_shuffles in zip(*training_data):
                self.log_debug('#'*80 + '\n', 'Checking out puzzle\n', puzzle, ' with nb_shuffles: ', nb_shuffles)
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
                        self.log_debug('  -> one away puzzle\n', one_away_puzzle,
                                       ', candidate_target: ', candidate_target)
                        one_away_puzzle_hash = hash(one_away_puzzle)
                        if one_away_puzzle_hash in hashes_to_puzzle:
                            candidate_target = min(candidate_target, hashes_to_puzzle[one_away_puzzle_hash][1])
                            self.log_debug('  -> is in hashes_to_puzzle candidate_target= ', candidate_target)
                        candidate_target += 1
                        target = min(target, candidate_target)
                        target = max(0, target)
                        self.log_debug('  -> target=', target)
                targets.append(target)
            targets = tensor(targets)
            puzzles_strings = [str(p) for p in puzzles]
            #puzzles = stack([puzzle.to_tensor().float().reshape(-1) for puzzle in puzzles])
            #for puzzle, target in zip(puzzles, targets):
            #    self.log_debug('puzzle:\n', puzzle, ', target: ', target,
            #                   'current net: ', self.current_network(puzzle))
            y_hat = self.current_network.evaluate(puzzles)
            loss = self.loss_function(y_hat, targets)
            self.log_info('epoch %d. loss: ' % epoch, loss.item())
            self.current_network.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if 0 == epoch % self.update_target_network_frequency:
                self.log_info('Updating target network')
                self.log_info('\n',
                              {'puzzle': puzzles_strings,
                               'y_hat': list(y_hat.flatten().tolist()),
                               'target': list(targets.flatten().tolist())})
                self.target_network = self.current_network.clone()
                
            
    def solve_impl(self, puzzle, time_out, **kw_args):
        return 0


########################################################################################################################