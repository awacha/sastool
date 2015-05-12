'''
Created on Jul 25, 2012

@author: andris
'''

import matplotlib.pyplot as plt
import numbers
import time
import matplotlib

__all__ = ['pause']

class Pauser(object):
    """A general, state-retaining class for pausing in a similar way as Matlab(R)
    does. After instantiation the pause() method can be called
    """
    _is_enabled = True
    def pause(self, arg=None):
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
        elif arg is None or isinstance(arg, str):
            self.do_ui_pause(arg)
        else:
            raise NotImplementedError(arg)
    def get_pause_state(self):
        """Return the current pause state (enabled/disabled)"""
        return self._is_enabled
    def set_pause_state(self, state=True):
        """Enable/disable pausing"""
        self._is_enabled = bool(state)
    def do_ui_pause(self, prompt=None):
        """Make a UI pause without regard to the current pause state."""
        if prompt is None:
            prompt = 'Paused. Press ENTER to continue...'
        if not plt.get_fignums():  # empty list: no figures are open
            input(prompt)
        else:  # a figure is open
            if matplotlib.get_backend().upper().startswith('GTK'):
                title_before = plt.gcf().canvas.get_toplevel().get_title()
            elif matplotlib.get_backend().upper().startswith('TK'):
                title_before = plt.gcf().canvas.get_tk_widget().winfo_toplevel().wm_title()
            elif matplotlib.get_backend().upper().startswith('WX'):
                title_before = plt.gcf().canvas.GetTopLevelParent().GetTitle()
            elif matplotlib.get_backend().upper().startswith('QT'):
                title_before = str(plt.gcf().canvas.topLevelWidget().windowTitle())
            else:
                title_before = 'Figure %d' % plt.gcf().number
            while True:  # wait until a key is pressed. Blink the title meanwhile.
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
