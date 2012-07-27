import searchpath
import pathutils
import pauser
import utils
import easylsq
import basicfit

from searchpath import *
from pathutils import *
from pauser import *
from utils import *
from easylsq import *
from basicfit import *
HC = 12398.419 #Planck's constant times speed of light, in eV*Angstrom units

class SASException(Exception):
    "General exception class for the package `sastool`"
    pass

