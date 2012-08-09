'''
Created on Jul 25, 2012

@author: andris
'''
import numpy as np
import re
import random

def parse_list_from_string(s):
    if not isinstance(s, basestring):
        raise ValueError('argument should be a string, not ' + type(s))
    s = s.strip()
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
        s = s[1:-1]
    else:
        raise ValueError('argument does not look like a list')
    return [parse_number(x.strip()) for x in s.split(',')]

def normalize_listargument(arg):
    """Check if arg is an iterable (list, tuple, set, dict, np.ndarray, except
        string!). If not, make a list of it. Numpy arrays are flattened and
        converted to lists."""
    if isinstance(arg, np.ndarray):
        return arg.flatten()
    if isinstance(arg, basestring):
        return [arg]
    if isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, dict) or isinstance(arg, set):
        return list(arg)
    return [arg]

def parse_number(val):
    """Try to auto-detect the numeric type of the value. First a conversion to
    int is tried. If this fails float is tried, and if that fails too, unicode()
    is executed. If this also fails, a ValueError is raised.
    """
    funcs = [int, float, parse_list_from_string, unicode]
    if (val.strip().startswith("'") and val.strip().endswith("'")) or (val.strip().startswith('"') and val.strip().endswith('"')):
        return val[1:-1]
    for f in funcs:
        try:
            return f(val)
        except ValueError:  #eat exception
            pass
    raise ValueError(val)

def flatten_hierarchical_dict(original_dict, separator='.', max_recursion_depth=None):
    """Flatten a dict.

    Inputs
    ------
    original_dict: dict
        the dictionary to flatten
    separator: string, optional
        the separator item in the keys of the flattened dictionary
    max_recursion_depth: positive integer, optional
        the number of recursions to be done. None is infinte.

    Output
    ------
    the flattened dictionary

    Notes
    -----
    Each element of `original_dict` which is not an instance of `dict` (or of a
    subclass of it) is kept as is. The others are treated as follows. If
    ``original_dict['key_dict']`` is an instance of `dict` (or of a subclass of
    `dict`), a corresponding key of the form
    ``key_dict<separator><key_in_key_dict>`` will be created in
    ``original_dict`` with the value of
    ``original_dict['key_dict']['key_in_key_dict']``.
    If that value is a subclass of `dict` as well, the same procedure is
    repeated until the maximum recursion depth is reached.

    Only string keys are supported.
    """
    if max_recursion_depth is not None and max_recursion_depth <= 0:
        #we reached the maximum recursion depth, refuse to go further
        return original_dict
    if max_recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = max_recursion_depth - 1
    dict1 = {}
    for k in original_dict:
        if not isinstance(original_dict[k], dict):
            dict1[k] = original_dict[k]
        else:
            dict_recursed = flatten_hierarchical_dict(original_dict[k], separator, next_recursion_depth)
            dict1.update(dict([(k + separator + x, dict_recursed[x]) for x in dict_recursed]))
    return dict1

def re_from_Cformatstring_numbers(s):
    """Make a regular expression from the C-style format string."""
    return "^" + re.sub('%\+?\d*l?[diou]', '\d+', s) + "$"


def random_str(Nchars=6, randstrbase='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    """Return a random string of <Nchars> characters. Characters are sampled
    uniformly from <randstrbase>.
    """
    return ''.join([randstrbase[random.randint(0, len(randstrbase) - 1)] for i in xrange(Nchars)])