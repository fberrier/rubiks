########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from locale import format_string, LC_ALL, setlocale
import numpy as np
from pandas import DataFrame, Series
from sys import platform
from tabulate import tabulate
########################################################################################################################
setlocale(LC_ALL, '')
########################################################################################################################


g_not_a_pkl_file = 'not_a_file.pkl'

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
