########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from pandas import concat, DataFrame, Series
from time import time as snap
from torch import tensor, no_grad
from torch.nn import MSELoss
from torch.optim import RMSprop
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.learner import Learner
from rubiks.utils.utils import ms_format
########################################################################################################################


class DRL(Learner):
    """ This learner will learn the cost-to-go via deep reinforcement learning """

    def __init__(self,
                 puzzle_type,
                 nb_epochs,
                 nb_shuffles,
                 nb_sequences,
                 **kw_args):
        Learner.__init__(self, puzzle_type, **kw_args)
        self.nb_epochs = nb_epochs
        self.nb_shuffles = nb_shuffles
        self.nb_sequences = nb_sequences
        self.target_network = DeepLearning.factory(puzzle_type, **kw_args)
        self.current_network = self.target_network.clone()
        self.update_target_network_frequency = kw_args.pop('update_target_network_frequency', 100)
        self.loss_function = MSELoss()
        self.learning_rate = kw_args.pop('learning_rate', 1e-6)
        self.verbose = kw_args.pop('verbose', False)

    epoch_tag = 'epoch'
    loss_tag = 'loss'
    latency_tag = 'latency'
    min_target_tag = 'min_target'
    max_target_tag = 'max_target'
    updated_network_tag = 'updated_network'

    class Decision(Enum):
        TBD = 'TBD'
        GRADIENT_DESCENT = 'GRADIENT_DESCENT'
        UPDATE_TARGET_NET = 'UPDATE_TARGET_NET'
        STOP = 'STOP'

    def decision(self, convergence_data) -> Decision:
        if len(convergence_data) >= self.nb_epochs:
            decision = self.Decision.STOP
        elif 0 == len(convergence_data) % self.update_target_network_frequency:
            decision = self.Decision.UPDATE_TARGET_NET
        else:
            decision = self.Decision.GRADIENT_DESCENT
        convergence_data.loc[len(convergence_data) - 1,
                             self.__class__.updated_network_tag] = decision
        self.log_info(convergence_data.iloc[len(convergence_data) - 1])
        return decision

    def learn(self):
        cls = self.__class__
        convergence_data = DataFrame(columns=[cls.epoch_tag,
                                              cls.loss_tag,
                                              cls.latency_tag,
                                              cls.min_target_tag,
                                              cls.max_target_tag,
                                              cls.updated_network_tag])
        optimizer = RMSprop(self.current_network.parameters(),
                            lr=self.learning_rate)
        epoch = 0
        while True:
            epoch += 1
            latency = -snap()
            puzzles = self.puzzle_type.get_training_data(nb_shuffles=self.nb_shuffles,
                                                         nb_sequences=self.nb_sequences,
                                                         strict=True,
                                                         one_list=True,
                                                         **self.kw_args)
            targets = []
            known_values = {hash(puzzle): self.target_network.evaluate(puzzle) for puzzle in puzzles}
            for puzzle in puzzles:
                self.log_debug('#'*80 + '\n', 'Checking out puzzle', puzzle)
                if puzzle.is_goal():
                    target = 0
                else:
                    target = float('inf')
                    one_away_puzzles = [puzzle.apply(move) for move in puzzle.possible_moves()]
                    for one_away_puzzle in one_away_puzzles:
                        candidate_target = self.target_network.evaluate(one_away_puzzle)
                        self.log_debug('  -> one away puzzle', one_away_puzzle,
                                       ', candidate_target: ', candidate_target)
                        one_away_puzzle_hash = hash(one_away_puzzle)
                        if one_away_puzzle_hash in known_values:
                            candidate_target = min(candidate_target, known_values[one_away_puzzle_hash])
                            self.log_debug('  -> is in hashes_to_puzzle candidate_target= ', candidate_target)
                        candidate_target += 1
                        target = min(target, candidate_target)
                        target = max(0, target)
                        self.log_debug('  -> target=', target)
                targets.append(target)
            targets = tensor(targets).float()
            if self.verbose:
                puzzles_strings = [str(p) for p in puzzles]
            y_hat = self.current_network.evaluate(puzzles)
            loss = self.loss_function(y_hat, targets)
            latency += snap()
            self.current_network.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            convergence_data = concat([convergence_data,
                                       Series({cls.epoch_tag: '%d/%d' % (epoch, self.nb_epochs),
                                               cls.loss_tag: '%.2e' % loss.item(),
                                               cls.latency_tag: ms_format(latency),
                                               cls.min_target_tag: min(targets),
                                               cls.max_target_tag: max(targets),
                                               cls.updated_network_tag: self.Decision.TBD}).to_frame().transpose()],
                                      ignore_index=True)
            decision = self.decision(convergence_data)
            if self.Decision.STOP == decision:
                break
            if self.Decision.UPDATE_TARGET_NET == decision:
                if self.verbose:
                    self.log_debug('Updating target network')
                    data = DataFrame({'puzzle': puzzles_strings,
                                      'y_hat': list(y_hat.flatten().tolist()),
                                      'hash': list(hash(p) for p in puzzles),
                                      'target': list(targets.flatten().tolist())}).sort_values(['target', 'hash'])
                    self.log_debug('/n', data)
                self.log_info('Updating target network')
                self.target_network = self.current_network.clone()

    def save(self, model_file):
        self.current_network.save(model_file)
        self.log_info('Saved learner state in ', model_file)

########################################################################################################################
