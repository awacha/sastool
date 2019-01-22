# pylint: disable=W0401

from . import arithmetic
from . import basicfit
from . import easylsq
from . import errorvalue
from . import fitter
from . import matplotlib_scales
from . import pathutils
from . import searchpath
from . import utils
from .arithmetic import *
from .basicfit import *
from .easylsq import *
from .errorvalue import *
from .fitter import *
from .pathutils import *
from .searchpath import *
from .utils import *


class SASException(BaseException):
    "General exception class for the package `sastool`"
    pass

__all__ = ['arithmetic', 'basicfit', 'easylsq', 'errorvalue', 'pathutils',
           'searchpath', 'utils', 'fitter']

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
