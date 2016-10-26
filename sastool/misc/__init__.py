# pylint: disable=W0401

from . import arithmetic
from . import basicfit
from . import easylsq
from . import errorvalue
from . import fitter
from . import fitter
from . import matplotlib_scales
from . import numerictests
from . import pathutils
from . import pauser
from . import rc
from . import searchpath
from . import utils
from .arithmetic import *
from .basicfit import *
from .easylsq import *
from .errorvalue import *
from .fitter import *
from .numerictests import *
from .pathutils import *
from .pauser import *
from .rc import sastoolrc
from .searchpath import *
from .utils import *


class SASException(BaseException):
    "General exception class for the package `sastool`"
    pass

__all__ = ['arithmetic', 'basicfit', 'easylsq', 'errorvalue', 'pathutils',
           'pauser', 'rc', 'searchpath', 'utils', 'fitter']

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
