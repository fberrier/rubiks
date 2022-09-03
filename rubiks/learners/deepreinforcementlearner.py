########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from functools import partial
from itertools import cycle
from matplotlib import pyplot as plt
from math import inf
from multiprocessing import Pool
from os.path import isfile
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
from torch import cuda, tensor
from torch.nn import MSELoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.learner import Learner
from rubiks.utils.utils import ms_format, h_format, pformat, to_pickle, \
    get_model_file_name, snake_case, billion, isinf, number_format
########################################################################################################################


class DeepReinforcementLearner(Learner):
    """ This learner will learn the cost-to-go via deep reinforcement learning
    """

    """ tags """
    epoch = 'epoch'
    loss = 'loss'
    loss_over_max_target = 'loss_over_max_target'
    latency = 'latency'
    min_target = 'min_target'
    max_target = 'max_target'
    max_max_target = 'max_max_target'
    target_network_count = 'target_network_count'
    network_name = 'network_name'
    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    decision = 'decision'
    cuda = 'cuda'
    puzzles_seen_pct = 'puzzles_seen_pct'

    """ config """
    nb_epochs = 'nb_epochs'
    nb_shuffles = 'nb_shuffles'
    min_no_loop = 'min_no_loop'
    nb_sequences = 'nb_sequences'
    update_target_network_frequency = 'update_target_network_frequency'
    default_update_target_network_frequency = 100
    update_target_network_threshold = 'update_target_network_threshold'
    default_update_target_network_threshold = 0.002
    max_nb_target_network_update = 'max_nb_target_network_update'
    default_max_nb_target_network_update = 100
    max_target_not_increasing_epochs_pct = 'max_target_not_increasing_epochs_pct'
    default_max_target_not_increasing_epochs_pct = 0.25
    max_target_uptick = 'max_target_uptick'
    default_max_target_uptick = 0.01
    use_cuda = 'use_cuda'
    learning_rate = 'learning_rate'
    optimiser = 'optimiser'
    rms_prop = 'rms_prop'
    scheduler = 'scheduler'
    no_scheduler = 'none'
    gamma_scheduler = 'gamma_scheduler'
    exponential_scheduler = 'exponential_scheduler'
    default_learning_rate = 1e-6
    nb_cpus = 'nb_cpus'
    default_nb_cpus = 1
    plot_metrics = 'plot_metrics'
    default_plot_metrics = [learning_rate,
                            #target_network_count,
                            max_target,
                            loss,
                            loss_over_max_target,
                            puzzles_seen_pct]
    training_data_every_epoch = 'training_data_every_epoch'
    cap_target_at_network_count = 'cap_target_at_network_count'

    @classmethod
    def populate_parser_impl(cls, parser):
        DeepLearning.populate_parser(parser)
        cls.add_argument(parser,
                         field=cls.nb_epochs,
                         type=int)
        cls.add_argument(parser,
                         field=cls.nb_shuffles,
                         type=int)
        cls.add_argument(parser,
                         field=cls.min_no_loop,
                         type=int)
        cls.add_argument(parser,
                         field=cls.nb_sequences,
                         type=int)
        cls.add_argument(parser,
                         field=cls.update_target_network_frequency,
                         type=int,
                         default=cls.default_update_target_network_frequency)
        cls.add_argument(parser,
                         field=cls.update_target_network_threshold,
                         type=float,
                         default=cls.default_update_target_network_threshold)
        cls.add_argument(parser,
                         field=cls.max_nb_target_network_update,
                         type=int,
                         default=cls.default_max_nb_target_network_update)
        cls.add_argument(parser,
                         field=cls.max_target_not_increasing_epochs_pct,
                         type=float,
                         default=cls.default_max_target_not_increasing_epochs_pct)
        cls.add_argument(parser,
                         field=cls.max_target_uptick,
                         type=float,
                         default=cls.default_max_target_uptick)
        cls.add_argument(parser,
                         field=cls.learning_rate,
                         type=float,
                         default=cls.default_learning_rate)
        cls.add_argument(parser,
                         field=cls.use_cuda,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.nb_cpus,
                         type=int,
                         default=cls.default_nb_cpus)
        cls.add_argument(parser,
                         field=cls.scheduler,
                         type=str,
                         choices=[cls.no_scheduler, cls.exponential_scheduler],
                         default=cls.no_scheduler)
        cls.add_argument(parser,
                         field=cls.optimiser,
                         type=str,
                         choices=[cls.rms_prop],
                         default=cls.rms_prop)
        cls.add_argument(parser,
                         field=cls.gamma_scheduler,
                         type=float,
                         default=0.99)
        cls.add_argument(parser,
                         field=cls.plot_metrics,
                         type=str,
                         nargs='+',
                         default=cls.default_plot_metrics)
        cls.add_argument(parser,
                         field=cls.training_data_every_epoch,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.cap_target_at_network_count,
                         default=False,
                         action=cls.store_true)

    def more_model_name(self):
        return list()

    def get_model_name(self):
        if isinstance(self.nb_shuffles, (int, float)) and not isinf(self.nb_shuffles):
            nb_shuffles = ['%dshf' % int(self.nb_shuffles)]
        elif isinf(self.nb_shuffles):
            nb_shuffles = []
        else:
            assert False, 'Unknown type for nb_shuffles [%s]' % self.nb_shuffles
        details = '_'.join([snake_case(self.__class__.__name__),
                            self.optimiser,
                            self.scheduler,
                            '%dgamma' % int(self.gamma_scheduler * billion),
                            '%dseq' % int(self.nb_sequences),
                            *nb_shuffles,
                            '%depc' % int(self.nb_epochs),
                            'tng_' + ('epoch' if self.training_data_every_epoch else 'ntk_updt'),
                            *self.more_model_name(),
                            ])
        if hasattr(self, 'cap_target_at_network_count') and self.cap_target_at_network_count:
            details += '_captgt'
        network_details = self.current_network.get_model_details()
        return get_model_file_name(self.get_puzzle_type(),
                                   self.get_puzzle_dimension(),
                                   model_name=details + '_' + network_details)

    class Decision(Enum):
        TBD = 'TBD'
        GRADIENT_DESCENT = 'GRADIENT_DESCENT'
        TARGET_NET_REGULAR_UPDATE = 'TARGET_NET_REGULAR_UPDATE'
        TARGET_NET_CONVERGENCE_UPDATE = 'TARGET_NET_CONVERGENCE_UPDATE'
        STOP = 'STOP'

    latency_tag = 'latency'
    latency_epoch_tag = 'epoch'
    latency_training_data_tag = 'training data'
    latency_target_data_tag = 'target data'
    latency_evaluate_tag = 'evaluate'
    latency_loss_tag = 'loss'
    latency_back_prop_tag = 'back prop'

    def recover_latency(self, latency_data=None):
        cls = self.__class__
        self.epoch_latency = latency_data[cls.latency_epoch_tag] if latency_data is not None else 0
        self.training_data_latency = latency_data[cls.latency_training_data_tag] if latency_data is not None else 0
        self.target_data_latency = latency_data[cls.latency_target_data_tag] if latency_data is not None else 0
        self.evaluate_latency = latency_data[cls.latency_evaluate_tag] if latency_data is not None else 0
        self.loss_latency = latency_data[cls.latency_loss_tag] if latency_data is not None else 0
        self.back_prop_latency = latency_data[cls.latency_back_prop_tag] if latency_data is not None else 0

    def read_data_file(self):
        data = read_pickle(self.learning_file_name)
        if self.config_tag not in data.keys():
            """ try copy """
            data = read_pickle(self.learning_file_name + '.copy')
        return data

    @classmethod
    def get_loss_function(cls):
        return MSELoss()

    def __init__(self, **kw_args):
        Learner.__init__(self, **kw_args)
        self.epoch_latency = None
        self.training_data_latency = None
        self.target_data_latency = None
        self.evaluate_latency = None
        self.loss_latency = None
        self.back_prop_latency = None
        # if the max value of target not increasing in that many epochs (as % of total epochs) by more than uptick
        # not much point going on
        self.use_cuda = self.use_cuda and cuda.is_available()
        self.loss_function = self.get_loss_function()
        cls = self.__class__
        self.attempt_recovery = self.learning_file_name is not None and isfile(self.learning_file_name)
        if self.attempt_recovery:
            try:
                data = self.read_data_file()
                config = data[self.config_tag]
                Learner.__init__(self, **config)
                self.convergence_data = data[self.convergence_data_tag]
                if self.convergence_data.empty:
                    raise RuntimeError('No convergence data to recover from')
                update_decisions = [cls.Decision.TARGET_NET_REGULAR_UPDATE,
                                    cls.Decision.TARGET_NET_CONVERGENCE_UPDATE]
                self.last_network_update = self.convergence_data[self.convergence_data.decision.
                    isin(update_decisions)][cls.epoch].iloc[-1]
                self.puzzles_seen = data[self.puzzles_seen_tag]
                self.target_network = DeepLearning.restore(data[self.network_data_tag])
                self.current_network = self.target_network.clone()
                self.recover_latency(data[self.latency_tag])
                self.target_network_count = 0
                self.log_info('Recovering from convergence_data: ', self.convergence_data.iloc[-1])
            except Exception as error:
                self.log_error('Could not recover from \'%s\': ' % self.learning_file_name,
                               error)
                self.attempt_recovery = False
        if not self.min_no_loop:
            self.min_no_loop = self.nb_shuffles
        self.pool_size = self.nb_cpus
        if self.attempt_recovery:
            return
        self.target_network = DeepLearning.factory(**kw_args)
        self.current_network = self.target_network.clone()
        self.convergence_data = DataFrame(columns=[cls.epoch,
                                                   cls.nb_epochs,
                                                   cls.puzzles_seen_pct,
                                                   cls.loss,
                                                   cls.learning_rate,
                                                   cls.latency,
                                                   cls.min_target,
                                                   cls.max_target,
                                                   cls.max_max_target,
                                                   cls.target_network_count,
                                                   cls.network_name,
                                                   cls.puzzle_type,
                                                   cls.puzzle_dimension,
                                                   cls.decision,
                                                   cls.nb_shuffles,
                                                   cls.nb_sequences,
                                                   cls.cuda])
        self.last_network_update = 0
        self.recover_latency()
        self.puzzles_seen = set()
        self.target_network_count = 0

    def get_decision(self, convergence_data) -> Decision:
        n = len(convergence_data)
        cls = self.__class__
        top = convergence_data.iloc[n - 1]
        stop = False
        if n >= self.nb_epochs:
            self.log_info('Reached max epochs')
            stop = True
        if not stop:
            new_max = top[cls.max_max_target]
            old_epoch = n - int(self.max_target_not_increasing_epochs_pct * self.nb_epochs)
            if old_epoch >= 0:
                old_max = convergence_data[cls.max_max_target][old_epoch]
                old_max *= (1 + self.max_target_uptick)
                if new_max <= old_max:
                    self.log_info('Max target not going up anymore')
                    stop = True
        if stop:
            decision = self.Decision.STOP
        elif n - self.last_network_update >= self.update_target_network_frequency:
            decision = self.Decision.TARGET_NET_REGULAR_UPDATE
            self.last_network_update = n
            if top[cls.target_network_count] == self.max_nb_target_network_update - 1:
                self.log_info('Reached max number of target network updates')
                decision = self.Decision.STOP
        elif abs(top[cls.loss] / top[cls.max_max_target]) <= self.update_target_network_threshold:
            decision = self.Decision.TARGET_NET_CONVERGENCE_UPDATE
            self.last_network_update = n
            if top[cls.target_network_count] == self.max_nb_target_network_update - 1:
                self.log_info('Reached max number of target network updates')
                decision = self.Decision.STOP
        else:
            decision = self.Decision.GRADIENT_DESCENT
        convergence_data.loc[n - 1, cls.decision] = decision
        top = convergence_data.iloc[n - 1]
        self.log_info(top)
        total_run_time = self.epoch_latency / top[cls.epoch]
        total_run_time *= top[cls.nb_epochs] - top[cls.epoch]
        self.log_info('Estimated max run time left: ', h_format(total_run_time),
                      '. Convergence update at loss <= %.2g' % (top[cls.max_max_target] *
                                                                self.update_target_network_threshold),
                      '. Regular update in ',
                      number_format(self.last_network_update + self.update_target_network_frequency - n),
                      ' epochs')
        return decision

    def __construct_target__(self, known_values, puzzle):
        try:
            if puzzle.is_goal():
                target = 0
            else:
                target = inf
                for move in puzzle.possible_moves():
                    one_away_puzzle = puzzle.apply(move)
                    candidate_target = self.target_network.evaluate(one_away_puzzle).item()
                    one_away_puzzle_hash = hash(one_away_puzzle)
                    if one_away_puzzle_hash in known_values:
                        candidate_target = min(candidate_target, known_values[one_away_puzzle_hash])
                    candidate_target += move.cost()
                    target = max(0, min(target, candidate_target))
        except KeyboardInterrupt:
            return None
        if self.cap_target_at_network_count:
            target = min(target, self.target_network_count)
        return target

    def get_optimiser_and_scheduler(self):
        if self.optimiser == self.rms_prop:
            optimizer = RMSprop
        else:
            raise NotImplementedError('Unknown optimiser [%s]' % self.optimiser)
        optimizer = optimizer(self.current_network.parameters(),
                              lr=float(self.learning_rate))
        scheduler = None if self.scheduler == self.no_scheduler else \
            ExponentialLR(optimizer, gamma=float(self.gamma_scheduler))
        return optimizer, scheduler

    @staticmethod
    def get_min_target(targets):
        return min(targets).item()

    @staticmethod
    def get_max_target(targets):
        return max(targets).item()

    def learn(self):
        cls = self.__class__
        interrupted = False
        try:
            optimizer, scheduler = self.get_optimiser_and_scheduler()
            epoch = 0 if self.convergence_data.empty else self.convergence_data[cls.epoch].iloc[-1]
            self.target_network_count = 1 if self.convergence_data.empty else \
                self.convergence_data[cls.target_network_count].iloc[-1]
            pool = Pool(self.pool_size)
            best_current = (inf, self.current_network.clone())
            puzzle_class = self.get_puzzle_type_class()
            config = self.get_config()
            possible_puzzles_nb = self.possible_puzzles_nb()
            decision = None
            while True:
                epoch += 1
                self.epoch_latency -= snap()
                self.training_data_latency -= snap()
                generate_new_training_data = self.training_data_every_epoch or \
                                             epoch == 1 or \
                                             decision in [self.Decision.TARGET_NET_REGULAR_UPDATE,
                                                          self.Decision.TARGET_NET_CONVERGENCE_UPDATE] or \
                                             self.attempt_recovery
                self.attempt_recovery = False
                if generate_new_training_data:
                    self.log_info('Generate ... ')
                    puzzles = puzzle_class.get_training_data(one_list=True, **config)
                    self.log_info(' ... ', len(puzzles), ' training puzzles')
                self.training_data_latency += snap()
                self.target_data_latency -= snap()
                if generate_new_training_data:
                    hashes = [hash(puzzle) for puzzle in puzzles]
                    self.puzzles_seen.update(hashes)
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
                    targets = tensor(targets).float()
                self.target_data_latency += snap()
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
                if scheduler:
                    scheduler.step()
                loss = loss.item()
                if loss < best_current[0]:
                    best_current = (loss, self.current_network.clone())
                min_targets = self.get_min_target(targets)
                max_targets = self.get_max_target(targets)
                old_max_targets = 0 if self.convergence_data.empty else \
                    self.convergence_data[cls.max_max_target].iloc[-1]
                max_max_targets = max(max_targets, old_max_targets)
                latency = Series({cls.latency_epoch_tag: ms_format(self.epoch_latency/epoch),
                                  cls.latency_training_data_tag: ms_format(self.training_data_latency/epoch),
                                  cls.latency_target_data_tag: ms_format(self.target_data_latency/epoch),
                                  cls.latency_evaluate_tag: ms_format(self.evaluate_latency/epoch),
                                  cls.latency_loss_tag: ms_format(self.loss_latency/epoch),
                                  cls.latency_back_prop_tag: ms_format(self.back_prop_latency/epoch),
                                  })
                latency = pformat(latency)
                learning_rate = self.learning_rate if scheduler is None else scheduler.get_last_lr()[0]
                puzzles_seen_pct = len(self.puzzles_seen) / possible_puzzles_nb * 100
                convergence_data = Series({cls.epoch: epoch,
                                           cls.nb_epochs: self.nb_epochs,
                                           cls.puzzles_seen_pct: puzzles_seen_pct,
                                           cls.loss: loss,
                                           cls.learning_rate: learning_rate,
                                           cls.latency: latency,
                                           cls.min_target: min_targets,
                                           cls.max_target: max_targets,
                                           cls.max_max_target: max_max_targets,
                                           cls.target_network_count: self.target_network_count,
                                           cls.puzzle_type: self.get_puzzle_type(),
                                           cls.puzzle_dimension: self.get_puzzle_dimension(),
                                           cls.network_name: self.target_network.get_name(),
                                           cls.decision: self.Decision.TBD,
                                           cls.nb_shuffles: self.nb_shuffles,
                                           cls.nb_sequences: self.nb_sequences,
                                           cls.cuda: self.use_cuda})
                convergence_data = convergence_data.to_frame()
                convergence_data = convergence_data.transpose()
                self.convergence_data = concat([self.convergence_data, convergence_data],
                                               ignore_index=True)
                decision = self.get_decision(self.convergence_data)
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
                    self.target_network_count += 1
                    optimizer, scheduler = self.get_optimiser_and_scheduler()
                    if self.learning_file_name:
                        self.save()
        except KeyboardInterrupt:
            self.log_warning('Was interrupted.')
            interrupted = True
        pool.close()
        pool.join()
        self.target_network = best_current[1].clone()
        if self.learning_file_name and not interrupted:
            self.save()

    network_data_tag = 'network_data'
    config_tag = 'config'
    convergence_data_tag = 'convergence_data'
    puzzles_seen_tag = 'puzzles_seen'

    def save(self):
        cls = self.__class__
        latency = {cls.latency_epoch_tag: self.epoch_latency,
                   cls.latency_training_data_tag: self.training_data_latency,
                   cls.latency_target_data_tag: self.target_data_latency,
                   cls.latency_evaluate_tag: self.evaluate_latency,
                   cls.latency_loss_tag: self.loss_latency,
                   cls.latency_back_prop_tag: self.back_prop_latency,
                   }
        data = {self.network_data_tag: self.current_network.get_data(),
                self.config_tag: self.get_config(),
                self.convergence_data_tag: self.convergence_data,
                self.puzzles_seen_tag: self.puzzles_seen,
                self.latency_tag: latency,
                }
        to_pickle(data, self.learning_file_name)
        count = self.target_network_count if self.__class__ is DeepReinforcementLearner else self.epoch
        versioned_learning_file_name = self.learning_file_name.replace('.pkl', '.%d.pkl' % count)
        to_pickle(data, versioned_learning_file_name)
        self.log_info('Saved learner state & convergence data in ',
                      self.learning_file_name,
                      ' and ',
                      versioned_learning_file_name)

    @staticmethod
    def axis_label_format(what):
        return what.upper().replace('_', ' ')

    def plot_learning(self):
        cls = self.__class__
        data = DeepReinforcementLearner.read_data_file(self)
        if isinstance(data[self.convergence_data_tag][cls.learning_rate].iloc[0], list):
            data[self.convergence_data_tag].loc[:, cls.learning_rate] = \
                data[self.convergence_data_tag][cls.learning_rate].apply(lambda lr: lr[0])
            to_pickle(data, self.learning_file_name)
            self.log_info('FIXED learning_rate -> ', self.learning_file_name)
        config = data[self.config_tag]
        learning_data = data[self.convergence_data_tag]
        drl = DeepReinforcementLearner  # alias
        puzzle_type = self.get_puzzle_type()
        puzzle_dimension = self.get_puzzle_dimension()
        data = learning_data[learning_data.puzzle_type == puzzle_type]
        data = data[data.puzzle_dimension == puzzle_dimension]
        assert 1 == len(set(data[drl.nb_shuffles]))
        assert 1 == len(set(data[drl.nb_sequences]))
        assert 1 == len(set(data[drl.network_name]))
        network_name = data[drl.network_name].iloc[0]
        title = {cls.puzzle_type: puzzle_type,
                 cls.puzzle_dimension: puzzle_dimension,
                 cls.network_name: network_name}
        fields_to_add = [cls.nb_sequences,
                         cls.nb_shuffles,
                         cls.learning_rate,
                         cls.optimiser]
        if config[cls.scheduler] != cls.no_scheduler:
            fields_to_add.extend([cls.scheduler, cls.gamma_scheduler])
        for field in fields_to_add:
            title[field] = config[field]
        title = pformat(title)
        title = cls.__name__ + '\n' + title
        fig = plt.figure(self.learning_file_name, figsize=(20, 10))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.axis('off')
        plt.title(title, fontname='Consolas')
        x = drl.epoch
        gs = fig.add_gridspec(len(self.plot_metrics), 1)
        axes = list()
        for a in range(len(self.plot_metrics)):
            axes.append(fig.add_subplot(gs[a]))
        colors = ['royalblue', 'orangered', 'seagreen', 'darkmagenta', 'gold']
        colors = cycle(colors)

        def add_plot(ax, y, c, x_label):
            if y in [drl.loss,
                     drl.loss_over_max_target,
                     drl.max_target,
                     drl.puzzles_seen_pct,
                     ]:
                label = '%s -> %.2g' % (y, data[y].iloc[-1])
            else:
                label = None
            ax.scatter(x,
                       y=y,
                       data=data,
                       color=c,
                       s=10,
                       marker='.',
                       label=label)
            if x_label:
                ax.set_xlabel(self.axis_label_format(x), fontweight='bold')
            ax.set_ylabel(self.axis_label_format(y), fontweight='bold')
            if label:
                ax.legend()
            if y in [drl.loss, drl.loss_over_max_target]:
                ax.set_yscale('log')
            ax.grid(True)

        data[drl.loss_over_max_target] = data[drl.loss] / data[drl.max_target]
        for a in range(len(self.plot_metrics)):
            add_plot(axes[a], self.plot_metrics[a], next(colors), x_label=a == len(self.plot_metrics) - 1)
        plt.tight_layout()
        plt.show()

########################################################################################################################
