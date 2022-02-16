########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from locale import format_string, LC_ALL, setlocale
setlocale(LC_ALL, '')
import numpy as np
from pandas import DataFrame, Series
from tabulate import tabulate
########################################################################################################################


def pformat(what):
    if isinstance(what, DataFrame):
        what = what.to_dict(orient='list')
    elif isinstance(what, Series):
        what = what.to_dict()
    if isinstance(what, dict):
        what = {k: [v] if not isinstance(v, (list, np.ndarray)) \
                else v for k, v in what.items()}
        return tabulate(what, headers='keys', tablefmt='psql')
    return what

########################################################################################################################


def pprint(what):
    """ util function to print tables nicely
    """
    print(pformat(what))

########################################################################################################################

        
def ms_format(run_time):
    run_time = run_time * 1000
    return '%s ms' % format_string('%d', run_time, grouping=True)
    
########################################################################################################################
