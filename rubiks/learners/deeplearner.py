########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from math import inf
from multiprocessing import Pool
from pandas import concat, DataFrame, Series
from time import time as snap
from torch import cuda, tensor
from torch.nn import MSELoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.heuristics.heuristic import Heuristic
from rubiks.learners.learner import Learner
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import ms_format, pformat, get_model_file_name
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
    default_plot_metrics = DeepReinforcementLearner.default_plot_metrics
    training_data_every_epoch = DeepReinforcementLearner.training_data_every_epoch
    threshold = 'threshold'

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
                         cls.threshold,
                         type=float,
                         default=0.001)

    def get_model_name(self):
        drl_details = '_'.join(['dl',
                                self.optimiser,
                                self.scheduler,
                                '%dseq' % int(self.nb_sequences),
                                '%dshf' % int(self.nb_shuffles),
                                '%depc' % int(self.nb_epochs),
                                'tng_' + ('epoch' if self.training_data_every_epoch else 'ntk_updt'),
                                ])
        network_details = self.target_network.get_model_details()
        return get_model_file_name(self.get_puzzle_type(),
                                   self.get_puzzle_dimension(),
                                   model_name=drl_details + '_' + network_details)

    Decision = DeepReinforcementLearner.Decision

    latency_tag = 'latency'
    latency_epoch_tag = 'epoch'
    latency_training_data_tag = 'training data'
    latency_target_data_tag = 'target data'
    latency_evaluate_tag = 'evaluate'
    latency_loss_tag = 'loss'
    latency_back_prop_tag = 'back prop'

    def __init__(self, **kw_args):
        Learner.__init__(self, **kw_args)
        # if the max value of target not increasing in that many epochs (as % of total epochs) by more than uptick
        # not much point going on
        self.use_cuda = self.use_cuda and cuda.is_available()
        self.loss_function = MSELoss()
        cls = self.__class__
        if not self.min_no_loop:
            self.min_no_loop = self.nb_shuffles
        self.pool_size = self.nb_cpus
        self.current_network = DeepLearning.factory(**kw_args)
        self.convergence_data = DataFrame(columns=[cls.epoch,
                                                   cls.nb_epochs,
                                                   cls.puzzles_seen_pct,
                                                   cls.loss,
                                                   cls.learning_rate,
                                                   cls.latency,
                                                   cls.min_target,
                                                   cls.max_target,
                                                   cls.max_max_target,
                                                   cls.network_name,
                                                   cls.puzzle_type,
                                                   cls.puzzle_dimension,
                                                   cls.decision,
                                                   cls.nb_shuffles,
                                                   cls.nb_sequences,
                                                   cls.cuda])
        self.epoch_latency = 0
        self.training_data_latency = 0
        self.target_data_latency = 0
        self.evaluate_latency = 0
        self.loss_latency = 0
        self.back_prop_latency = 0
        self.puzzles_seen = set()

    def get_decision(self, convergence_data) -> Decision:
        n = len(convergence_data)
        cls = self.__class__
        top = convergence_data.iloc[n - 1]
        stop = False
        if n >= self.nb_epochs:
            self.log_info('Reached max epochs')
            stop = True
        elif top[cls.loss] <= self.threshold:
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
                config[Solver.time_out] = inf
                config['plus'] = True
                target = Solver.factory(**config).solve(puzzle).cost
        except KeyboardInterrupt:
            return None
        return target

    def get_optimiser_and_scheduler(self):
        if self.optimiser == self.rms_prop:
            optimizer = RMSprop
        else:
            raise NotImplementedError('Unknown optimiser [%s]' % self.optimiser)
        optimizer = optimizer(self.current_network.parameters(),
                              lr=float(self.learning_rate))
        scheduler = None if self.scheduler == self.no_scheduler else \
            ExponentialLR(optimizer, gamma=self.gamma_scheduler)
        return optimizer, scheduler

    def learn(self):
        cls = self.__class__
        interrupted = False
        try:
            optimizer, scheduler = self.get_optimiser_and_scheduler()
            epoch = 0 if self.convergence_data.empty else self.convergence_data[cls.epoch].iloc[-1]
            pool = Pool(self.pool_size)
            puzzle_class = self.get_puzzle_type_class()
            config = self.get_config()
            possible_puzzles_nb = self.possible_puzzles_nb()
            while True:
                epoch += 1
                self.epoch_latency -= snap()
                self.training_data_latency -= snap()
                generate_new_training_data = self.training_data_every_epoch or 1 == epoch
                if generate_new_training_data:
                    puzzles = puzzle_class.get_training_data(one_list=True, **config)
                self.training_data_latency += snap()
                self.target_data_latency -= snap()
                if generate_new_training_data:
                    hashes = [hash(puzzle) for puzzle in puzzles]
                    self.puzzles_seen.update(hashes)
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
                    targets = tensor(targets).float()
                self.target_data_latency += snap()
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
                min_targets = min(targets).item()
                max_targets = max(targets).item()
                old_max_targets = 0 if self.convergence_data.empty else \
                    self.convergence_data[cls.max_target].iloc[-1]
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
                                           cls.puzzle_type: self.get_puzzle_type(),
                                           cls.puzzle_dimension: self.get_puzzle_dimension(),
                                           cls.network_name: self.current_network.get_name(),
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
        except KeyboardInterrupt:
            self.log_warning('Was interrupted.')
            interrupted = True
        pool.close()
        pool.join()
        if self.learning_file_name and not interrupted:
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
