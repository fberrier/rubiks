########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from itertools import product
from matplotlib import pyplot as plt
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
from torch import cuda, tensor
from torch.nn import MSELoss
from torch.optim import RMSprop
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.learner import Learner
from rubiks.utils.utils import ms_format, h_format
########################################################################################################################


class DeepReinforcementLearner(Learner):
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
        self.use_cuda = kw_args.pop('use_cuda', False) and cuda.is_available()
        self.loss_function = MSELoss()
        self.learning_rate = kw_args.pop('learning_rate', 1e-6)
        self.verbose = kw_args.pop('verbose', False)
        cls = self.__class__
        self.convergence_data = DataFrame(columns=[cls.epoch_tag,
                                                   cls.nb_epochs_tag,
                                                   cls.loss_tag,
                                                   cls.latency_tag,
                                                   cls.min_target_tag,
                                                   cls.max_target_tag,
                                                   cls.target_network_count_tag,
                                                   cls.network_name_tag,
                                                   cls.puzzle_type_tag,
                                                   cls.puzzle_dimension_tag,
                                                   cls.decision_tag,
                                                   cls.nb_shuffles_tag,
                                                   cls.nb_sequences_tag,
                                                   cls.cuda_tag])

    epoch_tag = 'epoch'
    loss_tag = 'loss'
    latency_tag = 'latency'
    min_target_tag = 'min_target'
    max_target_tag = 'max_target'
    target_network_count_tag = 'target_network_count'
    network_name_tag = 'network_name'
    puzzle_type_tag = 'puzzle_type'
    puzzle_dimension_tag = 'puzzle_dimension'
    decision_tag = 'decision'
    nb_shuffles_tag = 'nb_shuffles'
    nb_sequences_tag = 'nb_sequences'
    nb_epochs_tag = 'nb_epochs'
    cuda_tag = 'cuda'

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
                             self.__class__.decision_tag] = decision
        top = convergence_data.iloc[len(convergence_data) - 1]
        self.log_info(top)
        total_run_time = convergence_data[self.__class__.latency_tag].\
            apply(lambda l: int(l.replace('ms', '').replace(',', '')))
        total_run_time = sum(total_run_time)
        total_run_time = total_run_time / 1000 / top[self.__class__.epoch_tag]
        total_run_time *= top[self.__class__.nb_epochs_tag] - top[self.__class__.epoch_tag]
        self.log_info('Estimated run time left: ', h_format(total_run_time))
        return decision

    def learn(self):
        cls = self.__class__
        optimizer = RMSprop(self.current_network.parameters(),
                            lr=self.learning_rate)
        epoch = 0
        target_network_count = 1
        while True:
            epoch += 1
            latency = -snap()
            puzzles = self.puzzle_type.get_training_data(nb_shuffles=self.nb_shuffles,
                                                         nb_sequences=self.nb_sequences,
                                                         min_no_loop=self.nb_shuffles,
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
            if self.use_cuda:
                targets = targets.to()
            if self.verbose:
                puzzles_strings = [str(p) for p in puzzles]
            y_hat = self.current_network.evaluate(puzzles)
            if self.use_cuda and self.current_network.cuda_device:
                targets = targets.to(self.current_network.cuda_device)
            loss = self.loss_function(y_hat, targets)
            latency += snap()
            self.current_network.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            min_targets = min(targets).item()
            max_targets = max(targets).item()
            convergence_data = Series({cls.epoch_tag: epoch,
                                       cls.nb_epochs_tag: self.nb_epochs,
                                       cls.loss_tag: loss,
                                       cls.latency_tag: ms_format(latency),
                                       cls.min_target_tag: min_targets,
                                       cls.max_target_tag: max_targets,
                                       cls.target_network_count_tag: target_network_count,
                                       cls.puzzle_type_tag: self.get_puzzle_type(),
                                       cls.puzzle_dimension_tag: self.puzzle_dimension(),
                                       cls.network_name_tag: self.target_network.name(),
                                       cls.decision_tag: self.Decision.TBD,
                                       cls.nb_shuffles_tag: self.nb_shuffles,
                                       cls.nb_sequences_tag: self.nb_sequences,
                                       cls.cuda_tag: self.use_cuda})
            convergence_data = convergence_data.to_frame()
            convergence_data = convergence_data.transpose()
            self.convergence_data = concat([self.convergence_data, convergence_data],
                                           ignore_index=True)
            decision = self.decision(self.convergence_data)
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
                target_network_count += 1

    def save(self,
             model_file_name,
             learning_file_name=None):
        self.current_network.save(model_file_name)
        self.log_info('Saved learner state in ', model_file_name)
        if learning_file_name is not None:
            self.convergence_data.to_pickle(learning_file_name)
            self.log_info('Saved convergence data to ', learning_file_name)

    @staticmethod
    def plot_learning(learning_file_name,
                      network_name=None,
                      puzzle_type=None,
                      puzzle_dimension=None):
        learning_data = read_pickle(learning_file_name)
        drl = DeepReinforcementLearner # alias
        if network_name:
            learning_data = learning_data[learning_data.solver_name.apply(lambda sn: sn.find(network_name) >= 0)]
        if not puzzle_type:
            puzzle_types = set(learning_data[drl.puzzle_type_tag])
        else:
            puzzle_types = {puzzle_type}
        if not puzzle_dimension:
            puzzle_dimensions = set(learning_data[drl.puzzle_dimension_tag])
        else:
            puzzle_dimensions = {puzzle_dimension}
        for puzzle_type, puzzle_dimension in product(puzzle_types, puzzle_dimensions):
            data = learning_data[learning_data.puzzle_type == puzzle_type]
            data = data[data.puzzle_dimension == puzzle_dimension]
            assert 1 == len(set(data[drl.nb_shuffles_tag]))
            assert 1 == len(set(data[drl.nb_sequences_tag]))
            assert 1 == len(set(data[drl.network_name_tag]))
            if not network_name:
                network_name = data[drl.network_name_tag].iloc[0]
                network_name = network_name[:network_name.find('[')]
            title = '%s | %s | %s' % (network_name,
                                      puzzle_type.__name__,
                                      tuple(puzzle_dimension))
            fig = plt.figure(title)
            title = 'Learning data plot\n\n%s' % title
            fig.suptitle(title)
            x = drl.epoch_tag
            gs = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])

            def add_plot(ax, y, c):
                ax.scatter(x,
                           y=y,
                           data=data,
                           color=c)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                if drl.loss_tag == y:
                    ax.set_yscale('log')
            add_plot(ax1, drl.target_network_count_tag, 'b')
            add_plot(ax2, drl.max_target_tag, 'r')
            add_plot(ax3, drl.loss_tag, 'g')
        plt.show()

########################################################################################################################
