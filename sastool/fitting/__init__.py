'''sastool.fitting

Tools for fitting.

In this subpackage two class-hierarchies are defined:

FitFunction and descendants: various fitting functions. See help(FitFunction)
    for a general introduction.

Transform and descendants: transformations for sastool.dataset.PlotAndTransform
    and its descendants. Check out help(Transform)
'''
from . import fitfunctions
from . import standalone
from ..misc import easylsq

__all__ = ['fitfunctions', 'standalone', 'easylsq']

from .fitfunctions import *
from .standalone import *
from ..misc.easylsq import *

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
