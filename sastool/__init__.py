""" SASTool: a Python library for Small-angle scattering

Written by Andras Wacha (awacha at gmail dot com)

This library is intended for small-angle scatterers working at large-scale
facilities (synchrotron beamlines or neutrons) and at smaller laboratories. The
aim of this project is to create a toolchain for quick on-site data analysis,
data post-processing and fitting. I wrote (write) this with the requirement of
readability in mind, learning from an older project of mine (B1python, 
http://github.com/awacha/B1python). Much of the code in this has been simply
taken over from B1python and py_scidatatools, with refactoring, simplifying and
restructuring.

The library consists of the following parts:

io: input-output routines, for loading various raw and reduced measurement
    files.
dataset: a class-hierarchy for representing measured datasets and facilitating
    operations on them (fitting, arithmetics, plotting etc.)
misc: miscellaneous utility macros
utils2d: utilities for treating two-dimensional scattering data (integrating,
    finding the beam center, plotting etc.)
    
Please notice that this code is under development. The API is still subject to
changes, until the first major release (expected maybe in the second half of
2012). Please be patient (or help me... ;-))
"""

VERSION="0.0.1"
__all__=['io','misc','utils2d','dataset','fitting']

from . import misc
from . import utils2d
from . import io
from . import dataset
from . import fitting
