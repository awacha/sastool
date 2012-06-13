'''sastool.dataset

A versatile class-hierarchy for storage and operations on 1D data

End-user intended classes defined here are:

- `ErrorValue`: a value and its absolute error (SD)
- `DataSet`: class representing a typical measured dataset: abscissa, ordinate
  and errors of both. Various operations as plotting, fitting, saving, simple
  arithmetics are also available.
- `SASCurve`: Small-angle scattering curve. This is a specialized version of
  `DataSet`.

Mixin or interface classes:

- `ArithmeticBase`: support for basic arithmetic operations requiring minimal
  effort from the programmer
- `AliasedAttributes`: support for various __getattr__ tricks (i.e. alias names
  for attributes, attribute validation, default factory etc.)
- `AliasedArrayAttributes`: subclass to `AliasedAttributes` complete with a
  validating mechanism to ensure that all the controlled attributes are 
  ``numpy`` arrays of the same shape.
- `AliasedVectorAttributes`: a subclass to `AliasedArrayAttributes` where the
  controlled attributes are one-dimensional vectors. Four of the arrays have
  special meanings (those unaliasing to ``'_x'``, ``'_y'``, ``'_dy'`` and
  ``'_dx'``). Basic arithmetics is supported, as well as trimming, sorting,
  saving, interpolation, uniting, loading and calculating of momenta.
- `PlotAndTransform`: a mixin class enabling plotting and application of various
  transforms.
'''

__all__=['dataset','errorvalue','attributealias','arithmetic']

from dataset import DataSet, MatrixAttrMixin, AliasedVectorAttributes, PlotAndTransform, SASCurve, DataSetError, errtrapz
from errorvalue import ErrorValue
from attributealias import AliasedAttributes, AliasedArrayAttributes
from arithmetic import ArithmeticBase
