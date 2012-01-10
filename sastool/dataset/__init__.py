'''sastool.dataset

Implements a versatile class-hierarchy for storage and operations on scientific
datasets.

End-user intended classes defined here are:

ErrorValue: a value and its absolute error (SD)
DataSet: class representing a typical measured dataset: abscissa, ordinate and
    errors of both
SASCurve: Small-angle scattering curve (DataSet with different field names)

Mixin or interface classes:

ArithmeticBase: support for basic arithmetic operations requiring minimal effort
    from the programmer
AliasedAttributes: support for various __getattr__ tricks (i.e. alias names for
    different attributes, attribute validation, default creation)
AliasedArrayAttributes: descendant to AliasedAttributes complete with a
    validating mechanism to ensure that all the controlled attributes are numpy
    arrays of the same shape.
AliasedVectorAttributes: a descendant of AliasedArrayAttributes where the array
    attributes are one-dimensional vectors. Four of the arrays have special
    meanings (those unaliasing to '_x', '_y', '_dy', '_dx'). Basic arithmetics
    is supported, as well as trimming, sorting, saving, interpolation, uniting,
    loading and calculating of momenta.
PlotAndTransform: a mixin class enabling plotting and application of various
    transforms.
'''

__all__=['dataset','errorvalue','attributealias','arithmetic']

from dataset import DataSet, MatrixAttrMixin, AliasedVectorAttributes, PlotAndTransform, SASCurve, DataSetError, errtrapz
from errorvalue import ErrorValue
from attributealias import AliasedAttributes, AliasedArrayAttributes
from arithmetic import ArithmeticBase
