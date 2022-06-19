########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from functools import partial
from itertools import product
from matplotlib import pyplot as plt
from math import inf
from multiprocessing import Pool
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
from torch import cuda, tensor
from torch.nn import MSELoss
from torch.optim import RMSprop
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.learner import Learner
from rubiks.utils.utils import ms_format, h_format, pformat, touch
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
        self.log_info('Config=', kw_args)
        self.nb_epochs = nb_epochs
        self.nb_shuffles = nb_shuffles
        self.nb_sequences = nb_sequences
        self.target_network = DeepLearning.factory(puzzle_type, **kw_args)
        self.current_network = self.target_network.clone()
        self.update_target_network_frequency = kw_args.pop('update_target_network_frequency', 100)
        self.update_target_network_threshold = kw_args.pop('update_target_network_threshold', 0.01)
        self.max_nb_target_network_update = kw_args.pop('max_nb_target_network_update', 100)
        # if the max value of target not increasing in that many epochs (as % of total epochs) by more than uptick
        # not much point going on
        self.max_target_not_increasing_epochs_pct = kw_args.pop('max_target_not_increasing_epochs_pct', 0.1)
        self.max_target_uptick = kw_args.pop('max_target_uptick', 0.05)
        self.use_cuda = kw_args.pop('use_cuda', False) and cuda.is_available()
        self.loss_function = MSELoss()
        self.learning_rate = kw_args.pop('learning_rate', 1e-6)
        self.nb_cpus = kw_args.pop('nb_cpus', 1)
        self.verbose = kw_args.pop('verbose', False)
        cls = self.__class__
        self.convergence_data = DataFrame(columns=[cls.epoch_tag,
                                                   cls.nb_epochs_tag,
                                                   cls.loss_tag,
                                                   cls.latency_tag,
                                                   cls.min_target_tag,
                                                   cls.max_target_tag,
                                                   cls.max_max_target_tag,
                                                   cls.target_network_count_tag,
                                                   cls.network_name_tag,
                                                   cls.puzzle_type_tag,
                                                   cls.puzzle_dimension_tag,
                                                   cls.decision_tag,
                                                   cls.nb_shuffles_tag,
                                                   cls.nb_sequences_tag,
                                                   cls.cuda_tag])
        self.last_network_update = 0
        self.epoch_latency = 0
        self.training_data_latency = 0
        self.target_data_latency = 0
        self.evaluate_latency = 0
        self.loss_latency = 0
        self.back_prop_latency = 0
        self.pool_size = self.nb_cpus

    epoch_tag = 'epoch'
    loss_tag = 'loss'
    latency_tag = 'latency'
    min_target_tag = 'min_target'
    max_target_tag = 'max_target'
    max_max_target_tag = 'max_max_target'
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
        TARGET_NET_REGULAR_UPDATE = 'TARGET_NET_REGULAR_UPDATE'
        TARGET_NET_CONVERGENCE_UPDATE = 'TARGET_NET_CONVERGENCE_UPDATE'
        STOP = 'STOP'

    def decision(self, convergence_data) -> Decision:
        n = len(convergence_data)
        cls = self.__class__
        top = convergence_data.iloc[n - 1]
        stop = False
        if n >= self.nb_epochs:
            self.log_info('Reached max epochs')
            stop = True
        elif top[cls.target_network_count_tag] >= self.max_nb_target_network_update:
            self.log_info('Reached max number of target network updates')
            stop = True
        if not stop:
            new_max = top[cls.max_max_target_tag]
            old_epoch = n - int(self.max_target_not_increasing_epochs_pct * self.nb_epochs)
            if old_epoch >= 0:
                old_max = convergence_data[cls.max_max_target_tag][old_epoch]
                old_max *= (1 + self.max_target_uptick)
                if new_max <= old_max:
                    self.log_info('Max target not going up anymore')
                    stop = True
        if stop:
            decision = self.Decision.STOP
        elif n - self.last_network_update >= self.update_target_network_frequency:
            decision = self.Decision.TARGET_NET_REGULAR_UPDATE
            self.last_network_update = n
        elif abs(top[cls.loss_tag] / top[cls.max_max_target_tag]) <= self.update_target_network_threshold:
            decision = self.Decision.TARGET_NET_CONVERGENCE_UPDATE
            self.last_network_update = n
        else:
            decision = self.Decision.GRADIENT_DESCENT
        convergence_data.loc[n - 1, cls.decision_tag] = decision
        top = convergence_data.iloc[n - 1]
        self.log_info(top)
        total_run_time = self.epoch_latency / top[cls.epoch_tag]
        total_run_time *= top[cls.nb_epochs_tag] - top[cls.epoch_tag]
        self.log_info('Estimated max run time left: ', h_format(total_run_time),
                      '. Convergence update at loss <= %.4f' % (top[cls.max_max_target_tag] *
                                                                self.update_target_network_threshold),
                      '. Regular update in ',
                      self.last_network_update + self.update_target_network_frequency - n,
                      ' epochs')
        return decision

    def __construct_target__(self, known_values, puzzle):
        if puzzle.is_goal():
            target = 0
        else:
            target = inf
            one_away_puzzles = [puzzle.apply(move) for move in puzzle.possible_moves()]
            for one_away_puzzle in one_away_puzzles:
                candidate_target = self.target_network.evaluate(one_away_puzzle).item()
                one_away_puzzle_hash = hash(one_away_puzzle)
                if one_away_puzzle_hash in known_values:
                    candidate_target = min(candidate_target, known_values[one_away_puzzle_hash])
                candidate_target += 1
                target = min(target, candidate_target)
                target = max(0, target)
        return target

    def learn(self):
        cls = self.__class__
        optimizer = RMSprop(self.current_network.parameters(),
                            lr=self.learning_rate)
        epoch = 0
        target_network_count = 1
        pool = Pool(self.pool_size)
        best_current = (inf, self.current_network.clone())
        while True:
            epoch += 1
            self.epoch_latency -= snap()
            self.training_data_latency -= snap()
            puzzles = self.puzzle_type.get_training_data(nb_shuffles=self.nb_shuffles,
                                                         nb_sequences=self.nb_sequences,
                                                         min_no_loop=self.nb_shuffles,
                                                         one_list=True,
                                                         **self.kw_args)
            self.training_data_latency += snap()
            self.target_data_latency -= snap()
            hashes = [hash(puzzle) for puzzle in puzzles]
            values = self.target_network.evaluate(puzzles).squeeze().tolist()
            known_values = dict(zip(hashes, values))
            pool_size = min(self.nb_cpus, len(puzzles))
            if pool_size > 1:
                if pool_size != self.pool_size:
                    self.pool_size = pool_size
                    pool = Pool(pool_size)
                targets = pool.map(partial(self.__class__.__construct_target__,
                                           self,
                                           known_values),
                                   puzzles)
            else:
                targets = list(map(partial(self.__class__.__construct_target__,
                                           self,
                                           known_values),
                                   puzzles))
            self.target_data_latency += snap()
            targets = tensor(targets).float()
            if self.use_cuda:
                targets = targets.to()
            if self.verbose:
                puzzles_strings = [str(p) for p in puzzles]
            self.evaluate_latency -= snap()
            y_hat = self.current_network.evaluate(puzzles)
            self.evaluate_latency += snap()
            if self.use_cuda and self.current_network.cuda_device:
                targets = targets.to(self.current_network.cuda_device)
            self.loss_latency -= snap()
            loss = self.loss_function(y_hat, targets)
            self.loss_latency += snap()
            self.epoch_latency += snap()
            self.current_network.zero_grad()
            optimizer.zero_grad()
            self.back_prop_latency -= snap()
            loss.backward()
            self.back_prop_latency += snap()
            optimizer.step()
            loss = loss.item()
            if loss < best_current[0]:
                best_current = (loss, self.current_network.clone())
            min_targets = min(targets).item()
            max_targets = max(targets).item()
            old_max_targets = 0 if self.convergence_data.empty else \
                self.convergence_data[cls.max_target_tag].iloc[-1]
            max_max_targets = max(max_targets, old_max_targets)
            latency = Series({'epoch': ms_format(self.epoch_latency/epoch),
                              'training data': ms_format(self.training_data_latency/epoch),
                              'target data': ms_format(self.target_data_latency/epoch),
                              'evaluate': ms_format(self.evaluate_latency/epoch),
                              'loss': ms_format(self.loss_latency/epoch),
                              'back prop': ms_format(self.back_prop_latency/epoch),
                              })
            latency = pformat(latency)
            convergence_data = Series({cls.epoch_tag: epoch,
                                       cls.nb_epochs_tag: self.nb_epochs,
                                       cls.loss_tag: loss,
                                       cls.latency_tag: latency,
                                       cls.min_target_tag: min_targets,
                                       cls.max_target_tag: max_targets,
                                       cls.max_max_target_tag: max_max_targets,
                                       cls.target_network_count_tag: target_network_count,
                                       cls.puzzle_type_tag: self.get_puzzle_type().__name__,
                                       cls.puzzle_dimension_tag: str(tuple(self.puzzle_dimension())),
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
            if decision in [self.Decision.TARGET_NET_REGULAR_UPDATE,
                            self.Decision.TARGET_NET_CONVERGENCE_UPDATE]:
                if self.verbose:
                    self.log_debug('Updating target network')
                    data = DataFrame({'puzzle': puzzles_strings,
                                      'y_hat': list(y_hat.flatten().tolist()),
                                      'hash': list(hash(p) for p in puzzles),
                                      'target': list(targets.flatten().tolist())}).sort_values(['target', 'hash'])
                    self.log_debug('/n', data)
                self.log_info('Updating target network [%s]' % decision)
                self.target_network = best_current[1]
                best_current = (inf, self.target_network)
                target_network_count += 1
        pool.close()
        pool.join()

    def save(self,
             model_file_name,
             learning_file_name=None):
        touch(model_file_name)
        touch(learning_file_name)
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
            title = '%s | %s | %s' % (network_name,
                                      puzzle_type,
                                      puzzle_dimension)
            fig = plt.figure(title)
            title = 'Learning data plot\n\n%s' % title
            fig.suptitle(title)
            x = drl.epoch_tag
            gs = fig.add_gridspec(4, 1)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
            ax4 = fig.add_subplot(gs[3])
            loss_over_max_target_tag = 'loss_over_max_target'

            def add_plot(ax, y, c):
                ax.scatter(x,
                           y=y,
                           data=data,
                           color=c,
                           s=10,
                           marker='.')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                if y in [drl.loss_tag, loss_over_max_target_tag]:
                    ax.set_yscale('log')
            data[loss_over_max_target_tag] = data[drl.loss_tag] /  data[drl.max_target_tag]
            add_plot(ax1, drl.target_network_count_tag, 'b')
            add_plot(ax2, drl.max_target_tag, 'r')
            add_plot(ax3, drl.loss_tag, 'g')
            add_plot(ax4, loss_over_max_target_tag, 'm')
        plt.show()

########################################################################################################################
