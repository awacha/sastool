'miscellaneous utilities'
import numpy as np
import os
import sys
import numbers
import time
import matplotlib.pyplot as plt
import matplotlib
import random
import collections
import re
import stat

import fitting.easylsq

HC = 12398.419 #Planck's constant times speed of light, in eV*Angstrom units

global sastool_search_path

sastool_search_path = ['.']

class SASException(Exception):
    "General exception class for the package `sastool`"
    pass

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

def re_from_Cformatstring_numbers(s):
    """Make a regular expression from the C-style format string."""
    return "^" + re.sub('%\+?\d*l?[diou]', '\d+', s) + "$"

def findfileindirs(filename, dirs = None, use_pythonpath = True, use_searchpath = True, notfound_is_fatal = True, notfound_val = None):
    """Find file in multiple directories.
    
    Inputs:
        filename: the file name to be searched for.
        dirs: list of folders or None
        use_pythonpath: use the Python module search path
        use_searchpath: use the sastool search path.
        notfound_is_fatal: if an exception is to be raised if the file cannot be
            found. 
        notfound_val: the value which should be returned if the file is not
            found (only relevant if notfound_is_fatal is False)
    
    Outputs: the full path of the file.
    
    Notes:
        if filename is an absolute path by itself, folders in 'dir' won't be
            checked, only the existence of the file will be verified.
    """
    if os.path.isabs(filename):
        if os.path.exists(filename):
            return filename
        elif notfound_is_fatal:
            raise IOError('File ' + filename + ' not found.')
        else:
            return notfound_val
    if dirs is None:
        dirs = []
    dirs = normalize_listargument(dirs)
    if not dirs: #dirs is empty
        dirs = ['.']
    if use_pythonpath:
        dirs.extend(sys.path)
    if use_searchpath:
        global sastool_search_path
        dirs.extend(sastool_search_path)
    #expand ~ and ~user constructs
    dirs = [os.path.expanduser(d) for d in dirs]
    for d in dirs:
        if os.path.exists(os.path.join(d, filename)):
            return os.path.join(d, filename)
    if notfound_is_fatal:
        raise IOError('File %s not found in any of the directories.' % filename)
    else:
        return notfound_val

def energycalibration(energymeas, energycalib, energy1, degree = None):
    """Do energy calibration.
    
    Inputs:
        energymeas: vector of measured (apparent) energies
        energycalib: vector of theoretical energies corresponding to the measured ones
        energy1: vector or matrix or a scalar of apparent energies to calibrate.
        degree: degree of polynomial. If None, defaults to len(energymeas)-1.
        
    Output:
        the calibrated energy/energies, in the same form as energy1 was supplied
        
    Note:
        to do backward-calibration (theoretical -> apparent), swap energymeas
        and energycalib on the parameter list.
    """
    energymeas = normalize_listargument(energymeas)
    energycalib = normalize_listargument(energycalib)
    if degree is None:
        degree = len(energymeas) - 1
    if len(energymeas) == 1: # in this case, only do a shift.
        poly = [1, energycalib[0] - energymeas[0]]
    else: # if more energy values are given, do a linear fit.
        poly = np.lib.polynomial.polyfit(energymeas, energycalib, degree)
    return np.lib.polynomial.polyval(poly, energy1)

def parse_number(val):
    """Try to auto-detect the numeric type of the value. First a conversion to
    int is tried. If this fails float is tried, and if that fails too, unicode()
    is executed. If this also fails, a ValueError is raised.
    """
    funcs = [int, float, unicode]
    for f in funcs:
        try:
            return f(val)
        except ValueError:  #eat exception
            pass
    raise ValueError(val)

def flatten_hierarchical_dict(original_dict, separator = '.', max_recursion_depth = None):
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

class Pauser(object):
    """A general, state-retaining class for pausing in a similar way as Matlab(R)
    does. After instantiation the pause() method can be called
    """
    _is_enabled = True
    def pause(self, arg = None):
        """Make a pause or adjust pause mode, depending on the type and value of
        'arg':
        
        1) boolean (True/False): enable/disable pausing
        2) numeric: sleep for this many seconds
        3) string or None: Do an UI pause, i.e. if a matplotlib figure is open,
        wait until a key is pressed. If not, wait until a key is pressed in the
        command prompt. If the argument is a string, it will be used as the
        pause prompt. Does nothing if pausing is disabled.
        
        UI pausing depends on the currently used Matplotlib backend. Currently
        TK, GTK, WX and QT are tested.
        """
        if isinstance(arg, bool):
            self._is_enabled = arg
        elif isinstance(arg, numbers.Number):
            time.sleep(arg)
        elif arg is None or isinstance(arg, basestring):
            self.do_ui_pause(arg)
        else:
            raise NotImplementedError(arg)
    def get_pause_state(self):
        """Return the current pause state (enabled/disabled)"""
        return self._is_enabled
    def set_pause_state(self, state = True):
        """Enable/disable pausing"""
        self._is_enabled = bool(state)
    def do_ui_pause(self, prompt = None):
        """Make a UI pause without regard to the current pause state."""
        if prompt is None:
            prompt = 'Paused. Press ENTER to continue...'
        if not plt.get_fignums():   # empty list: no figures are open
            raw_input(prompt)
        else:  # a figure is open
            if matplotlib.get_backend().upper().startswith('GTK'):
                title_before = plt.gcf().canvas.get_toplevel().get_title()
            elif matplotlib.get_backend().upper().startswith('TK'):
                title_before = plt.gcf().canvas.get_tk_widget().winfo_toplevel().wm_title()
            elif matplotlib.get_backend().upper().startswith('WX'):
                title_before = plt.gcf().canvas.GetTopLevelParent().GetTitle()
            elif matplotlib.get_backend().upper().startswith('QT'):
                title_before = unicode(plt.gcf().canvas.topLevelWidget().windowTitle())
            else:
                title_before = u'Figure %d' % plt.gcf().number
            while True:  #wait until a key is pressed. Blink the title meanwhile.
                plt.gcf().canvas.set_window_title(prompt)
                result = plt.gcf().waitforbuttonpress(1)
                if result:  # waitforbuttonpress returns True for keypress, False for mouse click and None for timeout
                    break
                plt.gcf().canvas.set_window_title(title_before)
                result = plt.gcf().waitforbuttonpress(1)
                if result:  # waitforbuttonpress returns True for keypress, False for mouse click and None for timeout
                    break
            plt.gcf().canvas.set_window_title(title_before)


# create a "special" Pauser instance for convenience
_pauser = Pauser()
pause = _pauser.pause

def random_str(Nchars = 6, randstrbase = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    """Return a random string of <Nchars> characters. Characters are sampled
    uniformly from <randstrbase>.
    """
    return ''.join([randstrbase[random.randint(0, len(randstrbase) - 1)] for i in xrange(Nchars)])

def findpeak(x, y, dy = None, position = None, hwhm = None, baseline = None, amplitude = None):
    """Find a (positive) peak in the dataset.
    
    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        
    Outputs:
        peak position, error of peak position, hwhm, error of hwhm, baseline,
            error of baseline, amplitude, error of amplitude.
            
    Notes:
        A Gauss curve is fitted.
    """
    if position is None: position = x[y == y.max()]
    if hwhm is None: hwhm = 0.5 * (x.max() - x.min())
    if baseline is None: baseline = y.min()
    if amplitude is None: amplitude = y.max() - baseline
    if dy is None: dy = np.ones_like(x)
    def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
        return amplitude_ * np.exp(0.5 * (x_ - position_) ** 2 / hwhm_ ** 2) + baseline_
    p, dp = fitting.easylsq.nlsq_fit(x, y, dy, fitfunc,
                                     (amplitude, position, hwhm, baseline))[:2]
    return p[1], dp[1], abs(p[2]), dp[2], p[3], dp[3], p[0], dp[0]

def get_search_path():
    """Return the search path."""
    global sastool_search_path
    return sastool_search_path

def append_search_path(*folders):
    """Append folders to the search path."""
    global sastool_search_path
    if len(folders) == 1 and isinstance(folders[0], collections.Sequence):
        #allow for append_search_path(['dir1','dir2',...]) - like operation.
        folders = folders[0]
    sastool_search_path.extend([os.path.abspath(f) for f in folders])

def remove_from_search_path(folder):
    """Remove <folder> from the search path. Raise a ValueError if the entry is
    not on the search path."""
    global sastool_search_path
    abspath = os.path.abspath(folder)
    sastool_search_path.remove(abspath)

def set_search_path(pathlist):
    """Set the sastool file search path to <pathlist>."""
    global sastool_search_path
    sastool_search_path = pathlist

def find_subdirs(startdir = '.', recursion_depth = None):
    """Find all subdirectory of a directory.
    
    Inputs:
        startdir: directory to start with. Defaults to the current folder.
        recursion_depth: number of levels to traverse. None is infinite.
        
    Output: a list of absolute names of subfolders.
    
    Examples:
        >>> find_subdirs('dir',0)  # returns just ['dir']
        
        >>> find_subdirs('dir',1)  # returns all direct (first-level) subdirs
                                   # of 'dir'.
    """
    direct_subdirs = [os.path.join(startdir, x) for x in os.listdir(startdir) if stat.S_ISDIR(os.stat(os.path.join(startdir, x)).st_mode)]
    if recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = recursion_depth - 1
    if recursion_depth <= 1:
        return [startdir] + direct_subdirs
    else:
        subdirs = []
        for d in direct_subdirs:
            subdirs.extend(find_subdirs(d, next_recursion_depth))
        return [startdir] + subdirs

#    folder_slashes = os.path.abspath(os.path.expanduser(startdir)).count(os.sep)
#    return [x[0] for x in os.walk(startdir)
#        if os.path.abspath(x[0]).count(os.sep) - folder_slashes <= recursion_depth]

#def find_subdirs_old(startdir = '.', recursion_depth = np.inf):
#    """Find all subdirectory of a directory.
#    
#    Inputs:
#        startdir: directory to start with. Defaults to the current folder.
#        recursion_depth: number of levels to traverse. Default is infinite.
#        
#    Output: a list of absolute names of subfolders.
#    
#    Examples:
#        >>> find_subdirs('dir',0)  # returns just ['dir']
#        
#        >>> find_subdirs('dir',1)  # returns all direct (first-level) subdirs
#                                   # of 'dir'.
#    """
#    folder_slashes = os.path.abspath(os.path.expanduser(startdir)).count(os.sep)
#    return [x[0] for x in os.walk(startdir)
#        if os.path.abspath(x[0]).count(os.sep) - folder_slashes <= recursion_depth]
