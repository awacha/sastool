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
        return (max(vmin, max(minpos, 7 / 3 - 4 / 3 - 1)),
                max(vmax, max(minpos, 7 / 3 - 4 / 3 - 1)))

    class PowerTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, exponent):
            mtransforms.Transform.__init__(self)
            self.exponent = exponent

        def transform_non_affine(self, a):
            masked = ma.masked_where((a < 0), a)
            if masked.mask.any():
                return ma.power(a, self.exponent)
            else:
                return np.power(a, self.exponent)

        def inverted(self):
            return type(self)(1.0 / self.exponent)

mscale.register_scale(PowerScale)
