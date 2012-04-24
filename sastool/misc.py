'miscellaneous utilities'
import numpy as np
import os
import sys
import numbers
import time
import matplotlib.pyplot as plt
import matplotlib
import random

import fitting.easylsq

_search_path=['.']

def normalize_listargument(arg):
    """Check if arg is an iterable (list, tuple, set, dict, np.ndarray, except
        string!). If not, make a list of it. Numpy arrays are flattened and
        converted to lists."""
    if isinstance(arg,np.ndarray):
        return arg.flatten()
    if isinstance(arg,basestring):
        return [arg]
    if isinstance(arg,list) or isinstance(arg,tuple) or isinstance(arg,dict) or isinstance(arg,set):
        return list(arg)
    return [arg]

def findfileindirs(filename,dirs=[],use_pythonpath=True,notfound_is_fatal=True,notfound_val=None):
    """Find file in multiple directories."""
    if dirs is None:
        dirs=[]
    dirs=normalize_listargument(dirs)
    if not dirs: #dirs is empty
        dirs=['.']
    if use_pythonpath:
        dirs.extend(sys.path)
    #expand ~ and ~user constructs
    dirs=[os.path.expanduser(d) for d in dirs]
    for d in dirs:
        if os.path.exists(os.path.join(d,filename)):
            return os.path.join(d,filename)
    if notfound_is_fatal:
        raise IOError('File %s not found in any of the directories.' % filename)
    else:
        return notfound_val

def energycalibration(energymeas,energycalib,energy1,degree=None):
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
    energymeas=normalize_listargument(energymeas)
    energycalib=normalize_listargument(energycalib)
    if degree is None:
        degree=len(energymeas)-1
    if len(energymeas)==1: # in this case, only do a shift.
        poly=[1,energycalib[0]-energymeas[0]]
    else: # if more energy values are given, do a linear fit.
        poly=np.lib.polynomial.polyfit(energymeas,energycalib,degree)
    return np.lib.polynomial.polyval(poly,energy1)

def parse_number(val):
    """Try to auto-detect the numeric type of the value. First a conversion to
    int is tried. If this fails float is tried, and if that fails too, unicode()
    is executed. If this also fails, a ValueError is raised.
    """
    funcs=[int, float, unicode]
    for f in funcs:
        try:
            return f(val)
        except:
            pass
    raise ValueError(val)

class Pauser(object):
    """A general, state-retaining class for pausing in a similar way as Matlab(R)
    does. After instantiation the pause() method can be called
    """
    _is_enabled=True
    def pause(self,arg=None):
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
        if isinstance(arg,bool):
            self._is_enabled=arg
        elif isinstance(arg,numbers.Number):
            time.sleep(arg)
        elif arg is None or isinstance(arg,basestring):
            self.do_ui_pause(arg)
        else:
            raise NotImplementedError(arg)
    def get_pause_state(self):
        """Return the current pause state (enabled/disabled)"""
        return self._is_enabled
    def set_pause_state(self,state=True):
        """Enable/disable pausing"""
        self._is_enabled=bool(state);
    def do_ui_pause(self,prompt=None):
        """Make a UI pause without regard to the current pause state."""
        if prompt is None:
            prompt='Paused. Press ENTER to continue...'
        if not plt.get_fignums():   # empty list: no figures are open
            raw_input(prompt)
        else:  # a figure is open
            if matplotlib.get_backend().upper().startswith('GTK'):
                title_before=plt.gcf().canvas.get_toplevel().get_title()
            elif matplotlib.get_backend().upper().startswith('TK'):
                title_before=plt.gcf().canvas.get_tk_widget().winfo_toplevel().wm_title()
            elif matplotlib.get_backend().upper().startswith('WX'):
                title_before=plt.gcf().canvas.GetTopLevelParent().GetTitle();
            elif matplotlib.get_backend().upper().startswith('QT'):
                title_before=unicode(plt.gcf().canvas.topLevelWidget().windowTitle())
            else:
                title_before=u'Figure %d'%plt.gcf().number
            while True:  #wait until a key is pressed. Blink the title meanwhile.
                plt.gcf().canvas.set_window_title(prompt)
                result=plt.gcf().waitforbuttonpress(1);
                if result:  # waitforbuttonpress returns True for keypress, False for mouse click and None for timeout
                    break
                plt.gcf().canvas.set_window_title(title_before)
                result=plt.gcf().waitforbuttonpress(1);
                if result:  # waitforbuttonpress returns True for keypress, False for mouse click and None for timeout
                    break
            plt.gcf().canvas.set_window_title(title_before)


# create a "special" Pauser instance for convenience
_pauser=Pauser()
pause=_pauser.pause

_randstrbase='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def random_str(Nchars=6):
    return ''.join([_randstrbase[random.randint(0,len(_randstrbase)-1)] for i in xrange(Nchars)])

def findpeak(x,y,dy=None,position=None,hwhm=None,baseline=None,amplitude=None):
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
    if position is None: position=x[y==y.max()]
    if hwhm is None: hwhm=0.5*(x.max()-x.min())
    if baseline is None: baseline=y.min()
    if amplitude is None: amplitude=y.max()-baseline
    if dy is None: dy=np.ones_like(x)
    def fitfunc(x_,amplitude_,position_,hwhm_,baseline_):
        return amplitude_*np.exp(0.5*(x_-position_)**2/hwhm_**2)+baseline_
    p,dp,statdict=fitting.easylsq.nlsq_fit(x,y,dy,fitfunc,(amplitude,position,hwhm,baseline))
    return p[1],dp[1],abs(p[2]),dp[2],p[3],dp[3],p[0],dp[0]

def get_search_path():
    return _search_path

def append_search_path(folder):
    _search_path.append(os.path.abspath(folder))

def remove_from_search_path(folder):
    abspath=os.path.abspath(folder)
    _search_path.remove(abspath)
    
def set_search_path(pathlist):
    for k in _search_path:
        _search_path.remove(k)
    _search_path.extend(pathlist)
    
        