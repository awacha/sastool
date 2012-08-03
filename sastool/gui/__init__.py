"""GUI utilities

This package contains all the utilities which depend on a specific matplotlib
backend (pygtk). This is not imported by default with the main module.
"""

import maskmaker
import saspreview2d
import patheditor
import sasimagegui

from maskmaker import makemask
from patheditor import pathedit
from sasimagegui import *

def SAS2DGUI_run():
    w = sasimagegui.SASImageGuiMain()
    def f(widget, event, *args, **kwargs):    #IGNORE:W0613
        widget.destroy()
        del widget
    w.connect('delete-event', f)
    w.show()
