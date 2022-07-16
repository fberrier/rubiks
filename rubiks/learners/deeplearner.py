########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from math import inf
from multiprocessing import Pool
from numpy.random import randint
from os.path import isfile
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
from torch import cuda, tensor
from torch.nn import MSELoss
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.heuristics.heuristic import Heuristic
from rubiks.learners.learner import Learner
from rubiks.puzzle.trainingdata import TrainingData
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import ms_format, pformat
########################################################################################################################


class DeepLearner(Learner):
    """ This learner will learn the cost-to-go via deep learning
    """

    """ tags """
    epoch = 'epoch'
    loss = 'loss'
    loss_over_max_target = 'loss_over_max_target'
    latency = 'latency'
    min_target = 'min_target'
    max_target = 'max_target'
    max_max_target = 'max_max_target'
    network_name = 'network_name'
    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    decision = 'decision'
    cuda = 'cuda'
    puzzles_seen_pct = 'puzzles_seen_pct'

    """ config """
    nb_epochs = DeepReinforcementLearner.nb_epochs
    nb_shuffles = DeepReinforcementLearner.nb_shuffles
    nb_shuffles_min = 'nb_shuffles_min'
    nb_shuffles_max = 'nb_shuffles_max'
    min_no_loop = DeepReinforcementLearner.min_no_loop
    nb_sequences = DeepReinforcementLearner.nb_sequences
    use_cuda = DeepReinforcementLearner.use_cuda
    learning_rate = DeepReinforcementLearner.learning_rate
    optimiser = DeepReinforcementLearner.optimiser
    rms_prop = DeepReinforcementLearner.rms_prop
    scheduler = DeepReinforcementLearner.scheduler
    no_scheduler = DeepReinforcementLearner.no_scheduler
    gamma_scheduler = DeepReinforcementLearner.gamma_scheduler
    exponential_scheduler = DeepReinforcementLearner.exponential_scheduler
    default_learning_rate = DeepReinforcementLearner.default_learning_rate
    nb_cpus = DeepReinforcementLearner.nb_cpus
    default_nb_cpus = DeepReinforcementLearner.default_nb_cpus
    plot_metrics = DeepReinforcementLearner.plot_metrics
    default_plot_metrics = [learning_rate,
                            max_target,
                            loss,
                            loss_over_max_target,
                            puzzles_seen_pct]
    training_data_every_epoch = DeepReinforcementLearner.training_data_every_epoch
    training_data_freq = 'training_data_freq'
    threshold = 'threshold'
    save_at_each_epoch = 'save_at_each_epoch'
    high_target = 'high_target'
    training_data_from_data_base = 'training_data_from_data_base'

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
                         field=cls.nb_shuffles_min,
                         type=int)
        cls.add_argument(parser,
                         field=cls.nb_shuffles_max,
                         type=int)
        cls.add_argument(parser,
                         field=cls.min_no_loop,
                         type=int)
        cls.add_argument(parser,
                         field=cls.nb_sequences,
                         type=int)
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
                         field=cls.training_data_freq,
                         type=int,
                         default=0)
        cls.add_argument(parser,
                         field=cls.save_at_each_epoch,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.threshold,
                         type=float,
                         default=0.001)
        cls.add_argument(parser,
                         field=cls.high_target,
                         type=int,
                         default=0)
        cls.add_argument(parser,
                         field=cls.training_data_from_data_base,
                         default=False,
                         action=cls.store_true)

    def get_model_name(self):
        return DeepReinforcementLearner.get_model_name(self)

    def more_model_name(self):
        return ['%dminshf' % self.nb_shuffles_min,
                '%dmaxshf' % self.nb_shuffles_max]

    Decision = DeepReinforcementLearner.Decision

    latency_tag = DeepReinforcementLearner.latency_tag
    latency_epoch_tag = DeepReinforcementLearner.latency_epoch_tag
    latency_training_data_tag = DeepReinforcementLearner.latency_training_data_tag
    latency_target_data_tag = DeepReinforcementLearner.latency_target_data_tag
    latency_evaluate_tag = DeepReinforcementLearner.latency_evaluate_tag
    latency_loss_tag = DeepReinforcementLearner.latency_loss_tag
    latency_back_prop_tag = DeepReinforcementLearner.latency_back_prop_tag

    def __init__(self, **kw_args):
        Learner.__init__(self, **kw_args)
        if self.training_data_every_epoch:
            self.training_data_freq = 0
        if self.training_data_from_data_base:
            self.training_data = TrainingData(**self.get_config())
        assert self.training_data_freq >= 0, 'invalid training_data_freq'
        self.epoch_latency = None
        self.training_data_latency = None
        self.target_data_latency = None
        self.evaluate_latency = None
        self.loss_latency = None
        self.back_prop_latency = None
        # if the max value of target not increasing in that many epochs (as % of total epochs) by more than uptick
        # not much point going on
        self.use_cuda = self.use_cuda and cuda.is_available()
        self.loss_function = MSELoss()
        cls = self.__class__
        if not self.min_no_loop:
            self.min_no_loop = self.nb_shuffles
        if self.nb_shuffles_min > self.nb_shuffles_max:
            self.nb_shuffles_max, self.nb_shuffles_min = self.nb_shuffles_min, self.nb_shuffles_max
        if self.nb_shuffles_min < 0:
            self.nb_shuffles_min = self.nb_shuffles
        if self.nb_shuffles_max < 0:
            self.nb_shuffles_max = self.nb_shuffles
        self.pool_size = self.nb_cpus
        self.attempt_recovery = self.learning_file_name is not None and isfile(self.learning_file_name)
        if self.attempt_recovery:
            try:
                data = read_pickle(self.learning_file_name)
                config = data[self.config_tag]
                Learner.__init__(self, **config)
                if self.training_data_from_data_base:
                    self.training_data = TrainingData(**self.get_config())
                self.convergence_data = data[self.convergence_data_tag]
                if self.convergence_data.empty:
                    raise RuntimeError('No convergence data to recover from')
                self.puzzles_seen = data[self.puzzles_seen_tag]
                self.learning_rate = self.convergence_data[cls.learning_rate].iloc[-1]
                self.current_network = DeepLearning.restore(data[self.network_data_tag])
                DeepReinforcementLearner.recover_latency(self, data[self.latency_tag])
                self.log_info('Recovering from convergence_data: ', self.convergence_data.iloc[-1])
            except Exception as error:
                self.log_error('Could not recover from \'%d\': ' % self.learning_file_name,
                               error)
                self.attempt_recovery = False
        if self.attempt_recovery:
            return
        self.current_network = DeepLearning.factory(**kw_args)
        self.convergence_data = DataFrame(columns=[cls.epoch,
                                                   cls.nb_epochs,
                                                   cls.puzzles_seen_pct,
                                                   cls.loss,
                                                   cls.learning_rate,
                                                   cls.latency,
                                                   cls.min_target,
                                                   cls.max_target,
                                                   cls.network_name,
                                                   cls.puzzle_type,
                                                   cls.puzzle_dimension,
                                                   cls.decision,
                                                   cls.nb_shuffles,
                                                   cls.nb_shuffles_min,
                                                   cls.nb_shuffles_max,
                                                   cls.nb_sequences,
                                                   cls.cuda])
        DeepReinforcementLearner.recover_latency(self)
        self.puzzles_seen = set()

    def get_decision(self, convergence_data) -> Decision:
        n = len(convergence_data)
        cls = self.__class__
        top = convergence_data.iloc[n - 1]
        stop = False
        if n >= self.nb_epochs:
            self.log_info('Reached max epochs')
            stop = True
        elif abs(top[cls.loss] / top[cls.max_target]) <= self.threshold:
            self.log_info('Reached threshold')
            stop = True
        if stop:
            decision = self.Decision.STOP
        else:
            decision = self.Decision.GRADIENT_DESCENT
        convergence_data.loc[n - 1, cls.decision] = decision
        top = convergence_data.iloc[n - 1]
        self.log_info(top)
        return decision

    def __construct_target__(self, puzzle):
        try:
            if puzzle.is_goal():
                target = 0
            else:
                """ We solve using A* and admissible """
                config = self.get_config()
                config[Solver.solver_type] = Solver.astar
                config[Heuristic.heuristic_type] = Heuristic.manhattan
                config['plus'] = True
                solution = Solver.factory(**config).solve(puzzle)
                if solution.success:
                    target = solution.cost
                else:
                    target = self.high_target
        except KeyboardInterrupt:
            return None
        return target

    def get_optimiser_and_scheduler(self):
        return DeepReinforcementLearner.get_optimiser_and_scheduler(self)

    def get_training_data(self):
        self.log_info('Get training data')
        puzzle_class = self.get_puzzle_type_class()
        config = self.get_config()
        if self.training_data_from_data_base:
            self.training_data_latency -= snap()
            self.target_data_latency -= snap()
            puzzles = list()
            targets = list()
            for _ in range(self.nb_sequences):
                solution = self.training_data.get(randint(self.nb_shuffles_min, self.nb_shuffles_max + 1))
                assert solution.success
                puzzle = solution.puzzle.clone()
                puzzles.append(puzzle)
                cost = solution.cost
                targets.append(cost)
                for move in solution.path:
                    puzzle = puzzle.apply(move)
                    cost -= 1
                    puzzles.append(puzzle)
                    targets.append(cost)
                    assert cost >= 0, 'WTF?'
            self.training_data_latency += snap()
            self.target_data_latency += snap()
        else:
            self.training_data_latency -= snap()
            puzzles = puzzle_class.get_training_data(one_list=True, **config)
            self.training_data_latency += snap()
            self.target_data_latency -= snap()
            pool_size = min(self.nb_cpus, len(puzzles))
            if pool_size > 1:
                if pool_size != self.pool_size:
                    self.pool_size = pool_size
                    pool = Pool(pool_size)
                targets = pool.map(partial(self.__class__.__construct_target__,
                                           self),
                                   puzzles)
            else:
                targets = list(map(partial(self.__class__.__construct_target__,
                                           self),
                                   puzzles))
            self.target_data_latency += snap()
        self.puzzles_seen.update(hash(puzzle) for puzzle in puzzles)
        targets = tensor(targets).float()
        return puzzles, targets

    def learn(self):
        cls = self.__class__
        interrupted = False
        try:
            optimizer, scheduler = self.get_optimiser_and_scheduler()
            epoch = 0 if self.convergence_data.empty else self.convergence_data[cls.epoch].iloc[-1]
            pool = Pool(self.pool_size)
            best_current = (inf, self.current_network.clone())
            possible_puzzles_nb = self.possible_puzzles_nb()
            while True:
                epoch += 1
                self.epoch_latency -= snap()
                generate_new_training_data = self.training_data_every_epoch or \
                                             1 == epoch or \
                                             self.attempt_recovery or \
                                             (self.training_data_freq > 0 and 0 == epoch % self.training_data_freq)
                self.attempt_recovery = False
                if generate_new_training_data:
                    puzzles, targets = self.get_training_data()
                self.evaluate_latency -= snap()
                y_hat = self.current_network.evaluate(puzzles)
                self.evaluate_latency += snap()
                if self.use_cuda and self.current_network.cuda_device:
                    targets = targets.to(self.current_network.cuda_device)
                self.loss_latency -= snap()
                loss = self.loss_function(y_hat, targets)
                if loss < best_current[0]:
                    best_current = (loss, self.current_network.clone())
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
                min_targets = min(targets).item()
                max_targets = max(targets).item()
                latency = Series({cls.latency_epoch_tag: ms_format(self.epoch_latency / epoch),
                                  cls.latency_training_data_tag: ms_format(self.training_data_latency / epoch),
                                  cls.latency_target_data_tag: ms_format(self.target_data_latency / epoch),
                                  cls.latency_evaluate_tag: ms_format(self.evaluate_latency / epoch),
                                  cls.latency_loss_tag: ms_format(self.loss_latency / epoch),
                                  cls.latency_back_prop_tag: ms_format(self.back_prop_latency / epoch),
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
                                           cls.puzzle_type: self.get_puzzle_type(),
                                           cls.puzzle_dimension: self.get_puzzle_dimension(),
                                           cls.network_name: self.current_network.get_name(),
                                           cls.decision: self.Decision.TBD,
                                           cls.nb_shuffles: self.nb_shuffles,
                                           cls.nb_shuffles_min: self.nb_shuffles_min,
                                           cls.nb_shuffles_max: self.nb_shuffles_max,
                                           cls.nb_sequences: self.nb_sequences,
                                           cls.cuda: self.use_cuda})
                convergence_data = convergence_data.to_frame()
                convergence_data = convergence_data.transpose()
                self.convergence_data = concat([self.convergence_data, convergence_data],
                                               ignore_index=True)
                decision = self.get_decision(self.convergence_data)
                if self.Decision.STOP == decision:
                    break
                if self.save_at_each_epoch and self.learning_file_name:
                    self.save()
        except KeyboardInterrupt:
            self.log_warning('Was interrupted.')
            interrupted = True
        pool.close()
        pool.join()
        if self.learning_file_name and not interrupted:
            self.current_network = best_current[1].clone()
            self.save()

    network_data_tag = 'network_data'
    config_tag = 'config'
    convergence_data_tag = 'convergence_data'
    puzzles_seen_tag = 'puzzles_seen'

    def save(self):
        DeepReinforcementLearner.save(self)

    def plot_learning(self):
        DeepReinforcementLearner.plot_learning(self)

########################################################################################################################
