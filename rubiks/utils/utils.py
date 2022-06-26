########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from locale import format_string, LC_ALL, setlocale
from math import isinf
import numpy as np
from os import makedirs, remove, getenv
from os.path import exists, dirname
from pandas import DataFrame, Series, to_pickle as pandas_to_pickle
from pathlib import Path
from re import sub
from sys import platform
from tabulate import tabulate
########################################################################################################################
setlocale(LC_ALL, '')
########################################################################################################################


g_not_a_pkl_file = 'not_a_file.pkl'

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


def get_file_name(puzzle_type,
                  dimension,
                  file_type,
                  extension='pkl',
                  name=None):
    data_folder = getenv('RUBIKSDATA')
    if not data_folder:
        data_folder = str(Path.home()) + '/rubiks/data'
    possible_file_types = ['models', 'perf', 'shuffles', 'training']
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


def get_perf_file_name(puzzle_type,
                       dimension,
                       extension='pkl'):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type='perf',
                         extension=extension,
                         name='perf')

########################################################################################################################


def get_training_file_name(puzzle_type,
                           dimension,
                           model_name,
                           extension='pkl'):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type='training',
                         extension=extension,
                         name=model_name)

########################################################################################################################


def get_model_file_name(puzzle_type,
                        dimension,
                        model_name,
                        extension='pkl'):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type='models',
                         extension=extension,
                         name=model_name)

########################################################################################################################


def get_shuffles_file_name(puzzle_type,
                           dimension,
                           extension='pkl'):
    return get_file_name(puzzle_type=puzzle_type,
                         dimension=dimension,
                         file_type='shuffles',
                         extension=extension,
                         name='shuffles')

########################################################################################################################


def pformat(what):
    if isinstance(what, int):
        what = format_string('%d', what, grouping=True)
    elif isinstance(what, DataFrame):
        what = what.to_dict(orient='list')
    elif isinstance(what, Series):
        what = what.to_dict()
    if isinstance(what, dict):
        what = {k: [v] if not isinstance(v, (list, np.ndarray)) \
                else v for k, v in what.items()}
        return '\n' + tabulate(what, headers='keys', tablefmt='psql')
    return what

########################################################################################################################


def pprint(*what):
    """ util function to print tables nicely
    """
    print(*[pformat(w) for w in what])

########################################################################################################################


def number_format(what):
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

