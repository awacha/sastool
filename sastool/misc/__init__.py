# pylint: disable=W0401

import searchpath
import pathutils
import pauser
import utils
import easylsq
import basicfit
import rc
import errorvalue
import arithmetic

from searchpath import *
from pathutils import *
from pauser import *
from utils import *
from easylsq import *
from basicfit import *
from errorvalue import *
from arithmetic import *
from rc import sastoolrc
HC = 12398.419 #Planck's constant times speed of light, in eV*Angstrom units

class SASException(Exception):
    "General exception class for the package `sastool`"
    pass

__all__ = ['arithmetic', 'basicfit', 'easylsq', 'errorvalue', 'pathutils',
           'pauser', 'rc', 'searchpath', 'utils']

for k in __all__[:]:
    __all__.extend(eval('%s.__all__' % k))
