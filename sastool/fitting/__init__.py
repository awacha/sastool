'''sastool.fitting

Tools for fitting.

In this subpackage two class-hierarchies are defined:

FitFunction and descendants: various fitting functions. See help(FitFunction)
    for a general introduction.

Transform and descendants: transformations for sastool.dataset.PlotAndTransform
    and its descendants. Check out help(Transform)
'''
import sasbasic
import saspolymer
import basic
import bilayers
import sasspecial
from ..misc import easylsq

__all__ = ['basic', 'bilayers', 'sasbasic', 'sasspecial', 'saspolymer', 'easylsq']

from sasbasic import *
from saspolymer import *
from basic import *
from bilayers import *
from sasspecial import *
from ..misc.easylsq import *

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
