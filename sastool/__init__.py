"""
======================================================
 SASTool: a Python package for small-angle scattering 
======================================================

Written by Andras Wacha (awacha at gmail dot com)

This library is intended for small-angle scatterers working at large-scale
facilities (synchrotron beamlines or neutrons) and at smaller laboratories. The
aim of this project is to create a toolchain for quick on-site data analysis,
data post-processing and fitting. I wrote (write) this with the requirement of
readability in mind, learning from an older project of mine (``B1python``, 
http://github.com/awacha/B1python). Much of the code in this has been simply
taken over from ``B1python`` and ``py_scidatatools``, with refactoring, simplifying
and restructuring.

The library consists of the following sub-packages:

- `io`: input-output routines, for loading various raw and reduced measurement
  files.
- `dataset`: a class-hierarchy for representing measured datasets and facilitating
  operations on them (fitting, arithmetics, plotting etc.)
- `misc`: miscellaneous utility macros
- `utils2d`: utilities for treating two-dimensional scattering data (integrating,
  finding the beam center, plotting etc.)
- `fitting`: framework for nonlinear least-squares fitting
- `gui`: Utilities requiring graphical user interface. GUI is implemented in ``pygtk``,
  this subpackage depends therefore on the GTK backend of ``matplotlib``, therefore
  this subpackage is not imported by default.
- `sim`: Simulation routines

Please notice that this code is under development. The API is still subject to
changes, until the first major release (expected maybe in the second half of
2012). Please be patient (or help me out... ;-))
"""

__docformat__ = "restructuredtext en"

from _version import __version__

__all__ = ['io', 'misc', 'utils2d', 'dataset', 'fitting', 'sim']

import warnings
import matplotlib
#check if GTK library is present

try:
    # if we can import gtk and gtk can be initialized, we use that backend of 
    # matplotlib
    import gtk
    gtk.init_check() # returns None if OK, raises an exception (RuntimeError) if not.
    try:
        matplotlib.use('GTKAgg', warn = False)
    except TypeError:
        # older versions of IPython monkey-patch matplotlib.use(). That patch does not
        # support the 'warn' keyword argument
        matplotlib.use('GTKAgg')
except ImportError:
    # if gtk could not be imported, we still can use matplotlib but with a different backend
    warnings.warn('could not import gtk, GUI utilities won\'t work.')
except RuntimeError: # raised if we are on a terminal without a graphic display
    # gtk found but a graphical display has not been found: don't force GTKAgg
    # backend of matplotlib, as off-gui plotting (file-writing backends) can
    # work.
    warnings.warn('could not initialize graphic display, plotting won\'t work.')

import misc
import utils2d
import io
import dataset
import fitting
import sim

def _sas2dgui_main_program():
    """Entry point for the `sas2dutil` GUI script."""
    import gui
    a = gui.saspreview2d.SAS2DGUI()
    def delete_handler(*args, **kwargs):
        gtk.main_quit()
    a.connect('delete-event', delete_handler)
    a.show_all()
    gtk.main()
