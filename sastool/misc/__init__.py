# pylint: disable=W0401

from . import searchpath
from . import pathutils
from . import pauser
from . import utils
from . import easylsq
from . import basicfit
from . import rc
from . import errorvalue
from . import arithmetic
from . import numerictests
from . import matplotlib_scales

from .searchpath import *
from .pathutils import *
from .pauser import *
from .utils import *
from .easylsq import *
from .basicfit import *
from .errorvalue import *
from .arithmetic import *
from .numerictests import *
from .rc import sastoolrc

class SASException(BaseException):
    "General exception class for the package `sastool`"
    pass

__all__ = ['arithmetic', 'basicfit', 'easylsq', 'errorvalue', 'pathutils',
           'pauser', 'rc', 'searchpath', 'utils']

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
