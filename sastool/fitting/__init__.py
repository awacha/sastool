'''sastool.fitting

Tools for fitting.

In this subpackage two class-hierarchies are defined:

FitFunction and descendants: various fitting functions. See help(FitFunction)
    for a general introduction.

Transform and descendants: transformations for sastool.dataset.PlotAndTransform
    and its descendants. Check out help(Transform)
'''
from fitfunction import *
from transform import Transform
import sasbasic
import transform
import saspolymer
import basic
import bilayers
