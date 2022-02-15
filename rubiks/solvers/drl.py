########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from torch import tensor, stack
from torch.nn import MSELoss
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
        self.update_target_network_frequency = kw_args.pop('update_target_network_frequency', 10)
        self.loss_function = MSELoss(reduction='sum')

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
                self.log_info('#'*80 + '\n', 'Checking out puzzle\n', puzzle, ' with nb_shuffles: ', nb_shuffles)
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
                        self.log_info('  -> one away puzzle\n', one_away_puzzle,
                                      ', candidate_target: ', candidate_target)
                        one_away_puzzle_hash = hash(one_away_puzzle)
                        if one_away_puzzle_hash in hashes_to_puzzle:
                            candidate_target = min(candidate_target, hashes_to_puzzle[one_away_puzzle_hash][1])
                            self.log_info('  -> is in hashes_to_puzzle candidate_target= ', candidate_target)
                        candidate_target += 1
                        target = min(target, candidate_target)
                        target = max(0, target)
                        self.log_info('  -> target=', target)
                targets.append(target)
            targets = tensor(targets)
            puzzles = stack([puzzle.to_tensor().float().reshape(-1) for puzzle in puzzles])
            for puzzle, target in zip(puzzles, targets):
                self.log_info('puzzle:\n', puzzle, ', target: ', target,
                              'current net: ', self.current_network(puzzle))
            self.log_info('Let us run a forward backward epoch')
            y_hat = self.current_network(puzzles)
            loss = self.loss_function(y_hat, targets)
            self.log_info('loss: ', loss.item())
            self.current_network.zero_grad()
            loss.backward()

            #with torch.no_grad():
            #    for param in model.parameters():
            #        param -= learning_rate * param.grad

            
            self.log_info('y_hat=', y_hat)
            if 0 == epoch % self.update_target_network_frequency:
                self.target_network = self.current_network.clone()
                self.log_info('Updating target network')
                

    def solve_impl(self, puzzle, time_out, **kw_args):
        return 0


########################################################################################################################
