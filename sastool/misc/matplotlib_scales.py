from numpy import ma
import numpy as np

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import AutoLocator, ScalarFormatter, NullFormatter

class PowerScale(mscale.ScaleBase):
    """Scales data by raising it to a given power.
    """
    name = 'power'
    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
        self.exponent = kwargs.pop("exponent", 2)

    def get_transform(self):
        return self.PowerTransform(self.exponent)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 0), vmax

    class PowerTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, exponent):
            mtransforms.Transform.__init__(self)
            self.exponent = exponent

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the Mercator scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            masked = ma.masked_where((a < 0), a)
            if masked.mask.any():
                return ma.power(a, self.exponent)
            else:
                return np.power(a, self.exponent)

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return type(self)(1.0 / self.exponent)

mscale.register_scale(PowerScale)
        
