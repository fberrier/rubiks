########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from locale import format_string, LC_ALL, setlocale
from math import isinf, isnan
import numpy as np
from os import makedirs, remove, getenv, listdir
from os.path import exists, dirname, isfile, isdir, join
from pandas import DataFrame, Series, to_pickle as pandas_to_pickle, read_pickle
from pathlib import Path
from re import sub
from sys import platform
from tabulate import tabulate
########################################################################################################################
setlocale(LC_ALL, '')
########################################################################################################################


thousand = 1000
million = 1000000
billion = 1000000000

########################################################################################################################


def remove_file(file_name):
    try:
        remove(file_name)
    except FileNotFoundError:
        pass


########################################################################################################################


def touch(file_name):
    if file_name is not None and not exists(dirname(file_name)):
        makedirs(dirname(file_name))

########################################################################################################################


def to_pickle(what, file_name):
    touch(file_name)
    pandas_to_pickle(what, file_name)

########################################################################################################################


def snake_case(s):
    return '_'.join(sub('([A-Z][a-z]+)', r' \1',
                    sub('([A-Z]+)', r' \1',
                    s.replace('-', ' '))).split()).lower()

########################################################################################################################


def from_snake_case(s):
    return ''.join([s.capitalize() for s in s.split('_')])

########################################################################################################################


class Extension:
    pkl = 'pkl'

########################################################################################################################


class PossibleFileNames:
    models = 'models'
    perf = 'perf'
    shuffles = 'shuffles'
    training = 'training'
    manhattan = 'manhattan'
    utils = 'utils'
    all = [models,
           perf,
           shuffles,
           training,
           manhattan,
           utils,
           ]

########################################################################################################################


def get_file_name(puzzle_type,
                  dimension,
                  file_type,
                  extension=Extension.pkl,
                  name=None):
    data_folder = getenv('RUBIKSDATA')
    if not data_folder:
        data_folder = str(Path.home()) + '/rubiks/data'
    possible_file_types = PossibleFileNames.all
    assert file_type in possible_file_types, 'Unknown file_type [%s]. Choose from %s' % (file_type, possible_file_types)
    assert name, 'Empty name'
    if isinstance(puzzle_type, type):
        puzzle_type = snake_case(puzzle_type.__name__)
    if not isinstance(dimension, tuple):
        dimension = tuple(dimension)
    dimension = '_'.join((str(d) for d in dimension))
    extension = '.%s' % (extension.replace('.', ''))
    fn = '/'.join([data_folder, file_type, puzzle_type, dimension, name]) + extension
    fn = fn.replace('//', '/').replace('\\', '/')
    return fn

########################################################################################################################


def get_performance_file_name(puzzle_type,
                              dimension,
                              extension=Extension.pkl):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type=PossibleFileNames.perf,
                         extension=extension,
                         name=PossibleFileNames.perf)

########################################################################################################################


def get_training_file_name(puzzle_type,
                           dimension,
                           extension=Extension.pkl):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type=PossibleFileNames.training,
                         extension=extension,
                         name=PossibleFileNames.training)

########################################################################################################################


def get_model_file_name(puzzle_type,
                        dimension,
                        model_name,
                        extension=Extension.pkl):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type=PossibleFileNames.models,
                         extension=extension,
                         name=model_name)

########################################################################################################################


def get_shuffles_file_name(puzzle_type,
                           dimension,
                           extension=Extension.pkl):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type=PossibleFileNames.shuffles,
                         extension=extension,
                         name=PossibleFileNames.shuffles)

########################################################################################################################


def pformat(what, floatfmt=None, showindex=False):
    if floatfmt is None:
        floatfmt = '.2g'
    if isinstance(what, int):
        what = format_string('%d', what, grouping=True)
    elif isinstance(what, DataFrame):
        what = what.to_dict(orient='list')
        what = {k: [v] if not isinstance(v, (list, np.ndarray)) \
                else v for k, v in what.items()}
        return '\n' + tabulate(DataFrame(what),
                               headers='keys',
                               floatfmt=floatfmt,
                               tablefmt='psql',
                               showindex=showindex)
    elif isinstance(what, Series):
        what = what.to_dict()
        what = {k: ('%.2g' % v) if isinstance(v, float) else v for k, v in what.items()}
    if isinstance(what, dict):

        def massage(i):
            if not isinstance(i, (list, np.ndarray)):
                return [i]
            if isinstance(i, (list, np.ndarray)):
                return str(i)
            return i
        what = {k: massage(v) for k, v in what.items()}
        try:
            what = DataFrame(what)
        except ValueError:
            what = DataFrame(what, index=[0])
        return '\n' + tabulate(what.transpose(),
                               showindex=True,
                               floatfmt='.2g',
                               tablefmt='psql')
    return what

########################################################################################################################


def pprint(*what):
    """ util function to print tables nicely
    """
    print(*[pformat(w) for w in what])

########################################################################################################################


def number_format(what):
    if isnan(what):
        return 'nan'
    elif isinf(what):
        return 'inf'
    return format_string('%d',
                         int(what),
                         grouping=True)

########################################################################################################################

        
def ms_format(run_time):
    run_time = run_time * 1000
    return '%s ms' % format_string('%d', run_time, grouping=True)
    
########################################################################################################################


def s_format(run_time):
    return '%s s' % format_string('%d', run_time, grouping=True)

########################################################################################################################


def h_format(run_time):
    hour = int(run_time / 3600)
    minute = int((run_time - hour * 3600) / 60)
    return '%sh%sm' % (hour, minute)

########################################################################################################################


def hms_format(run_time):
    hour = int(run_time / 3600)
    minute = int((run_time - hour * 3600) / 60)
    second = int(run_time - hour * 3600 - minute * 60)
    return '%sh%sm%ss' % (hour, minute, second)

########################################################################################################################


def is_windows():
    return 'win32' == platform

########################################################################################################################


def is_mac_os():
    return 'darwin' == platform

########################################################################################################################


def is_linux():
    return platform in {'linux', 'linux2'}

########################################################################################################################


def is_inf(what):
    if isinstance(what, str):
        return what.lower() == 'inf'
    return isinf(what)

########################################################################################################################


def out_of_order(some_permutation):
    some_permutation = np.array(some_permutation)
    total = 0
    assert len(some_permutation) == len(set(some_permutation)), 'Duplicates in inputs'
    for index, value in enumerate(some_permutation):
        total += sum(some_permutation[index + 1:] < value)
    return total

########################################################################################################################


def bubble_sort_swaps_count(some_permutation):
    """ return number of swap operations of the bubble sort """
    some_permutation = np.array(some_permutation)
    assert len(some_permutation) == len(set(some_permutation)), 'Duplicates in inputs'
    if len(some_permutation) <= 1:
        return 0
    total = 0
    swapped = True
    while swapped:
        swapped = False
        for index in range(len(some_permutation) - 1):
            if some_permutation[index] > some_permutation[index + 1]:
                swapped = True
                total += 1
                left, right = some_permutation[index], some_permutation[index + 1]
                some_permutation[index], some_permutation[index + 1] = right, left
    return total

########################################################################################################################


def lighten_deep_reinforcement_learning_file(learning_file_name):
    from rubiks.core.loggable import Loggable
    logger = Loggable(name='lighten_deep_reinforcement_learning_file')
    if isdir(learning_file_name):
        logger.log_info('This is a directory. Will call function on all pkl files in here:')
        pkl_files = [pf for pf in listdir(learning_file_name) if
                     isfile(join(learning_file_name, pf)) and pf.endswith('.pkl')]
        for pkl_file in pkl_files:
            logger.log_info('\t -> ', join(learning_file_name, pkl_file))
            lighten_deep_reinforcement_learning_file(join(learning_file_name, pkl_file))
        return
    data = read_pickle(learning_file_name)
    from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
    safe = learning_file_name + '.copy'
    if isfile(safe):
        logger.log_info('Safety copy already exists, not over-writing as could lose data by doing so')
        return
    keys = [DeepReinforcementLearner.network_data_tag,
            DeepReinforcementLearner.config_tag,
            DeepReinforcementLearner.convergence_data_tag,
            DeepReinforcementLearner.puzzles_seen_tag,
            DeepReinforcementLearner.latency_tag]
    assert all(key in data.keys() for key in keys)
    to_pickle(data, safe)
    logger.log_info('Made a safety copy of \'%s\' into \'%s\'' % (learning_file_name, safe))
    keep = [DeepReinforcementLearner.network_data_tag,
            DeepReinforcementLearner.convergence_data_tag]
    data = {key: data[key] for key in keep}
    data[DeepReinforcementLearner.convergence_data_tag] = data[DeepReinforcementLearner.convergence_data_tag][-1:]
    to_pickle(data, learning_file_name)
    logger.log_info('Saved trimmed data back into \'%s\'' % learning_file_name)

########################################################################################################################


def fix_perf_learning_file(perf_file_name):
    """ I used to have a bug where what should have shown as exponential_scheduler was showing as
    gamma_scheduler ... this function fixes pkl files by replacing the former by the latter
    """
    from rubiks.core.loggable import Loggable
    logger = Loggable(name='fix_perf_learning_file')
    if isdir(perf_file_name):
        logger.log_info('This is a directory. Will call function on all pkl files in here:')
        pkl_files = [pf for pf in listdir(perf_file_name) if
                     isfile(join(perf_file_name, pf)) and pf.endswith('.pkl')]
        for pkl_file in pkl_files:
            logger.log_info('\t -> ', join(perf_file_name, pkl_file))
            fix_perf_learning_file(join(perf_file_name, pkl_file))
        return
    data = read_pickle(perf_file_name)
    solver_name_tag = 'solver_name'
    if not isinstance(data, DataFrame) or solver_name_tag not in data.columns:
        return
    from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
    data.loc[:, solver_name_tag] = data[solver_name_tag].apply\
        (lambda solver_name: solver_name.replace(DeepReinforcementLearner.gamma_scheduler,
                                                 DeepReinforcementLearner.exponential_scheduler))
    data.to_pickle(perf_file_name)
    logger.log_info('Replaced \'', DeepReinforcementLearner.gamma_scheduler, '\' by \'',
                    DeepReinforcementLearner.exponential_scheduler,
                    '\' in ', perf_file_name)

########################################################################################################################

