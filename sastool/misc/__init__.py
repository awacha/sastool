# pylint: disable=W0401

import searchpath
import pathutils
import pauser
import utils
import easylsq
import basicfit
import rc
import errorvalue

from searchpath import *
from pathutils import *
from pauser import *
from utils import *
from easylsq import *
from basicfit import *
from errorvalue import *
from rc import sastoolrc
HC = 12398.419 #Planck's constant times speed of light, in eV*Angstrom units

class SASException(Exception):
    "General exception class for the package `sastool`"
    pass

