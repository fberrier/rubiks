########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from locale import format_string, LC_ALL, setlocale
from math import isinf
import numpy as np
from os import makedirs
from os.path import exists, dirname
from pandas import DataFrame, Series
from pathlib import Path
from re import sub
from sys import platform
from tabulate import tabulate
########################################################################################################################
setlocale(LC_ALL, '')
########################################################################################################################


g_not_a_pkl_file = 'not_a_file.pkl'

########################################################################################################################


def touch(what):
    if what is not None and not exists(dirname(what)):
        makedirs(dirname(what))

########################################################################################################################


def snake_case(s):
    return '_'.join(sub('([A-Z][a-z]+)', r' \1',
                    sub('([A-Z]+)', r' \1',
                    s.replace('-', ' '))).split()).lower()

########################################################################################################################


def file_name(puzzle_type,
              dimension,
              file_type,
              extension='pkl',
              name=None):
    home = str(Path.home()) + '/rubiks/data'
    possible_file_types = ['models', 'perf', 'shuffles', 'training']
    assert file_type in possible_file_types, 'Unknown file_type [%s]. Choose from %s' % (file_type, possible_file_types)
    assert name, 'Empty name'
    if isinstance(puzzle_type, type):
        puzzle_type = snake_case(puzzle_type.__name__)
    if not isinstance(dimension, tuple):
        dimension = tuple(dimension)
    dimension = '_'.join((str(d) for d in dimension))
    extension = '.%s' % (extension.replace('.', ''))
    fn = '/'.join([home, file_type, puzzle_type, dimension, name]) + extension
    fn = fn.replace('//', '/').replace('\\', '/')
    return fn

########################################################################################################################


def perf_file_name(puzzle_type,
                   dimension,
                   extension='pkl'):
    return file_name(puzzle_type=puzzle_type,
                     dimension=dimension,
                     file_type='perf',
                     extension=extension,
                     name='perf')

########################################################################################################################


def training_file_name(puzzle_type,
                       dimension,
                       model_name,
                       extension='pkl'):
    return file_name(puzzle_type=puzzle_type,
                     dimension=dimension,
                     file_type='training',
                     extension=extension,
                     name=model_name)

########################################################################################################################


def model_file_name(puzzle_type,
                    dimension,
                    model_name,
                    extension='pkl'):
    return file_name(puzzle_type=puzzle_type,
                     dimension=dimension,
                     file_type='models',
                     extension=extension,
                     name=model_name)

########################################################################################################################


def shuffles_file_name(puzzle_type,
                       dimension,
                       extension='pkl'):
    return file_name(puzzle_type=puzzle_type,
                     dimension=dimension,
                     file_type='shuffles',
                     extension=extension,
                     name='shuffles')

########################################################################################################################


def pformat(what):
    if isinstance(what, DataFrame):
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

