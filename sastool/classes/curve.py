'''
Created on Jul 5, 2012

@author: andris
'''

import collections
import gzip
import itertools
import numbers
import operator
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from sastool.misc.arithmetic import ArithmeticBase
from sastool.misc.easylsq import nonlinear_leastsquares, simultaneous_nonlinear_leastsquares, nonlinear_odr
from sastool.misc.errorvalue import ErrorValue

__all__ = ['GeneralCurve', 'SASCurve', 'SASPixelCurve', 'SASAzimuthalCurve']


def errtrapz(x, yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable

    Outputs:
        the error of the integral
    """
    x = np.array(x)
    yerr = np.array(yerr)
    return 0.5 * np.sqrt((x[1] - x[0]) ** 2 * yerr[0] ** 2 +
                         np.sum((x[2:] - x[:-2]) ** 2 * yerr[1:-1] ** 2) +
                         (x[-1] - x[-2]) ** 2 * yerr[-1] ** 2)


class ControlledVectorAttribute(object):
    """A class exposing the descriptor interface (__get__ and __set__ methods)
    to be used as special arguments in GeneralCurve and its derivatives.
    """

    def __init__(self, value=None, name=None, obj=None):
        if isinstance(value, type(self)):
            self.value = value.value
            self.name = value.name
        else:
            self.name = name
            value_orig = value
            value = np.array(value)
            if value.dtype.kind.lower() not in 'biuf':
                raise TypeError("Cannot instantiate a ControlledVectorAttribute with a type %s; numpy dtype kind was %s." % (
                    type(value_orig), value.dtype.kind))
            self.value = obj.check_compatibility(value, self.name)

    def __get__(self, obj, type_=None):
        return self.value

    def __set__(self, obj, value):
        self.value = obj.check_compatibility(value, self.name)

    def __delete__(self, obj):
        if hasattr(obj, '_controlled_attributes'):
            obj._controlled_attributes.remove(self.name)


class GeneralCurve(ArithmeticBase):
    _xtolpcnt = 1e-2  # tolerance percent in abscissa (1e-2 corresponds to 1%)
    _dxepsilon = 1e-6
    _default_special_names = [
        ('x', 'X'), ('y', 'Y'), ('dy', 'DY'), ('dx', 'DX')]
    _lastaxes = None

    def __init__(self, *args, **kwargs):
        ArithmeticBase.__init__(self)
        kwargs = self._initialize_kwargs_for_readers(kwargs)
        self._controlled_attributes = []
        self._special_names = kwargs['special_names']
        if len(args) == 0:
            # empty constructor
            self.x = None
            self.y = None
            self.dy = None
            self.dx = None
        elif isinstance(args[0], type(self)):
            # copy constructor
            self._special_names = args[0]._special_names
            for name in args[0]._controlled_attributes:
                self.add_dataset(name, getattr(args[0], name))
        elif isinstance(args[0], dict):
            for k in args[0]:
                self.add_dataset(k, args[0][k])
        elif isinstance(args[0], str):
            kwargs_for_loadtxt = kwargs.copy()
            del kwargs_for_loadtxt['special_names']
            del kwargs_for_loadtxt['autodetect_columnnames']
            columnnames = itertools.chain(self._special_names.values(),
                                          ('column%d' % x for x in itertools.count(
                                              len(self._special_names)))
                                          )
            if kwargs['autodetect_columnnames']:
                with open(args[0], 'rt') as f:
                    print('.')
                    firstline = f.readline()
                    if firstline.startswith('#'):
                        columnnames = firstline[1:].strip().split()
            m = np.loadtxt(args[0], **kwargs_for_loadtxt)
            for i, cn in zip(list(range(m.shape[1])), columnnames):
                self.add_dataset(cn, m[:, i])
        else:
            for a, name in zip(args, list(self._special_names.values())):
                self.add_dataset(name, a)
        for a, name in [(kwargs[k], k) for k in kwargs if isinstance(kwargs[k], np.ndarray) or
                        (isinstance(kwargs[k], collections.Sequence) and all(
                            isinstance(x, numbers.Number) or isinstance(x, ErrorValue) for x in kwargs[k]))
                        or isinstance(kwargs[k], ErrorValue)]:
            self.add_dataset(name, a)

    def __getattribute__(self, name):
        # hack to allow instance descriptors.
        # http://blog.brianbeck.com/post/74086029/instance-descriptors
        value = object.__getattribute__(self, name)
        if isinstance(value, ControlledVectorAttribute):
            value = value.__get__(self, self.__class__)
        return value

    def __setattr__(self, name, value):
        # hack to allow instance descriptors.
        # http://blog.brianbeck.com/post/74086029/instance-descriptors
        if hasattr(self, '_special_names') and name in list(self._special_names.values()):
            self.add_dataset(name, value)
        try:
            obj = object.__getattribute__(self, name)
        except AttributeError:
            pass
        else:
            if isinstance(obj, ControlledVectorAttribute):
                return obj.__set__(self, value)
        return object.__setattr__(self, name, value)

    def __delattr__(self, name):
        obj = object.__getattribute__(self, name)
        if isinstance(obj, ControlledVectorAttribute):
            obj.__delete__(self)
        object.__delattr__(self, name)

    def _initialize_kwargs_for_readers(self, kwargs):
        if 'special_names' not in kwargs:
            kwargs['special_names'] = collections.OrderedDict(
                self._default_special_names)
        if 'autodetect_columnnames' not in kwargs:
            kwargs['autodetect_columnnames'] = False
        return kwargs

    def _get_xname(self): return self.get_specialname('x')

    def _set_xname(self, value): return self.set_specialname('x', value)

    def _get_yname(self): return self.get_specialname('y')

    def _set_yname(self, value): return self.set_specialname('y', value)

    def _get_dxname(self): return self.get_specialname('dx')

    def _set_dxname(self, value): return self.set_specialname('dx', value)

    def _get_dyname(self): return self.get_specialname('dy')

    def _set_dyname(self, value): return self.set_specialname('dy', value)

    xname = property(_get_xname, _set_xname, None, 'The name of the abscissa')
    yname = property(_get_yname, _set_yname, None, 'The name of the ordinate')
    dxname = property(
        _get_dxname, _set_dxname, None, 'The name of the error of the abscissa')
    dyname = property(
        _get_dyname, _set_dyname, None, 'The name of the error of the ordinate')

    def _get_x(self): return self.__getattribute__(self.xname)

    def _set_x(self, value): return self.add_dataset(self.xname, value)

    def _get_y(self): return self.__getattribute__(self.yname)

    def _set_y(self, value): return self.add_dataset(self.yname, value)

    def _get_dx(self): return self.__getattribute__(self.dxname)

    def _set_dx(self, value): return self.add_dataset(self.dxname, value)

    def _get_dy(self): return self.__getattribute__(self.dyname)

    def _set_dy(self, value): return self.add_dataset(self.dyname, value)
    x = property(_get_x, _set_x)
    y = property(_get_y, _set_y)
    dx = property(_get_dx, _set_dx)
    dy = property(_get_dy, _set_dy)

    def __gt__(self, other):
        return self.y > other

    def __lt__(self, other):
        return self.y < other

    def __eq__(self, other):
        return self.y == other

    def __ge__(self, other):
        return self.y >= other

    def __le__(self, other):
        return self.y <= other

    def __ne__(self, other):
        return self.y != other

    def set_specialname(self, specname, newname):
        """Set a new name to a special name

        Inputs:
        -------
            specname: string
                one of 'x', 'y', 'dx', 'dy'
            newname: string
                the new alias name (i.e. 'q' for 'x')
        """
        if newname == specname:
            raise ValueError('Name `%s` not supported for %s. Try `%s`.' % (
                newname, specname, specname.upper()))
        self._special_names[specname] = newname

    def get_specialname(self, specname):
        """Get the currently associated name for the special attribute `specname`.
        """
        return self._special_names[specname]

    def check_compatibility(self, value, name):
        """Check if `value` is allowed to be added as argument `name`.
        This currently checks if the length of `value` is the same as the length
        of the curve given by ``len(self)``.
        """
        try:
            l = len(self)
        except ValueError:
            return np.array(value)
        if value is None:
            self.remove_dataset(name)
        elif len(value) == l:
            return np.array(value)
        else:
            raise ValueError('Incompatible length!')

    def add_dataset(self, name, value):
        """Add a new controlled attribute (such as x, y, dy, dx or their alias
        names defined in self._special_names).

        Inputs:
        -------
            `name`: string
                the name of the new attribute. For default arguments (x, y, dy,
                dx) both the canonical name and the alias name can be given.
            `value`: various types are accepted
                The actual data set. Depending on the type:
                    `None`: the data set is removed, i.e. a call is issued to
                        ``self.remove_dataset(name)``
                    `ErrorValue`: two (!) controlled attributes are added: one
                        from the `.val` field and the other from `.err` field,
                        with the name `'d'+name` or `'D'+name`, the capitalization
                        depending on the first letter of `name`.
                    a sequence (instance of `collections.Sequence`) of `ErrorValue`
                        instances: works the same as if `value` would be an
                        instance of `ErrorValue`.
                    other: anything that can be converted to a np.ndarray of a
                        numeric type by `np.array()`
        """
        if value is None:
            return self.remove_dataset(name)
        if isinstance(value, ErrorValue):
            if name[0] == name[0].upper():
                errprefix = 'D'
            elif name[0] == name[0].lower():
                errprefix = 'd'
            retval = object.__setattr__(
                self, name, ControlledVectorAttribute(value.val, name, self))
            self._controlled_attributes.append(name)
            object.__setattr__(
                self, errprefix + name, ControlledVectorAttribute(value.err, 'd' + name, self))
            self._controlled_attributes.append(errprefix + name)
        elif isinstance(value, collections.Sequence) and all([isinstance(item, ErrorValue) for item in value]):
            return self.add_dataset(name, ErrorValue(value))
        else:
            retval = object.__setattr__(
                self, name, ControlledVectorAttribute(value, name, self))
            self._controlled_attributes.append(name)
        return retval

    def remove_dataset(self, name):
        """Remove the special dataset `name` if it is present"""
        if hasattr(self, name):
            self.__delattr__(name)

    def __len__(self):
        for name in ['x', 'y', 'dx', 'dy']:
            try:
                return len(self.__getattribute__(name))
            except AttributeError:
                pass
        raise ValueError('This SASClass has no length (no dataset yet)')

    def check_arithmetic_compatibility(self, other):
        if isinstance(other, ErrorValue):
            return other
        elif isinstance(other, numbers.Number):
            return ErrorValue(other)
        elif isinstance(other, np.ndarray):
            if len(other) == 1:
                return ErrorValue(other[0])
            elif len(other) == len(self):
                return ErrorValue(other.flatten())
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, type(self)):
            if not len(other) == len(self):
                raise ValueError('Incompatible shape!')
            if hasattr(other, 'dx') and hasattr(self, 'dx') and ((self.dx ** 2 + other.dx ** 2).mean() > max(self._dxepsilon, other._dxepsilon)):
                if (((self.dx ** 2 + other.dx ** 2) ** 0.5 - np.absolute(self.x - other.x)) <= 0).all():
                    return other
                else:
                    #raise ValueError('Abscissae are not the same within error!')
                    return other
            elif (np.absolute(other.x - self.x) <= np.absolute(other.x + self.x) * 0.5 * max(self._xtolpcnt, other._xtolpcnt)).all():
                return other
            else:
                #raise ValueError('Incompatible abscissae!')
                return other
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        try:
            other = self.check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        if isinstance(other, ErrorValue):
            self.y = self.y + other.val
            if not hasattr(self, 'dy'):
                self.dy = np.zeros_like(self.y)
            self.dy = np.sqrt(self.dy ** 2 + other.err ** 2)
        elif isinstance(other, GeneralCurve):
            self.x = 0.5 * (self.x + other.x)  # IGNORE:E1103
            self.y = self.y + other.y  # IGNORE:E1103
            if not hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = other.dx  # IGNORE:E1103
            elif hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = 0.5 * (self.dx + other.dx)  # IGNORE:E1103
            # the other two cases do not need explicit treatment.
            if not hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = other.dy  # IGNORE:E1103
            elif hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = np.sqrt(self.dy ** 2 + other.dy ** 2)  # IGNORE:E1103
        else:
            return NotImplemented
        return self

    def __imul__(self, other):
        try:
            other = self.check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        if isinstance(other, GeneralCurve):
            if not hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = other.dx
            elif hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = 0.5 * (self.dx + other.dx)
            # the other two cases do not need explicit treatment.
            if not hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = other.dy
            elif hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = np.sqrt(
                    self.dy ** 2 * other.y ** 2 + other.dy ** 2 * self.y ** 2)
            self.x = 0.5 * (self.x + other.x)
            self.y = self.y * other.y
        elif isinstance(other, ErrorValue):
            if not hasattr(self, 'dy'):
                self.dy = np.zeros_like(self.y)
            self.dy = np.sqrt(
                self.dy ** 2 * other.val ** 2 + other.err ** 2 * self.y ** 2)
            self.y = self.y * other.val
        else:
            return NotImplemented
        return self

    def __neg__(self):
        obj = type(self)(self)  # make a copy
        obj.y = -obj.y
        return obj

    def __reciprocal__(self):
        obj = type(self)(self)
        if hasattr(self, 'dy'):
            obj.dy = (1.0 * obj.dy) / (obj.y * obj.y)
        obj.y = 1.0 / obj.y
        return obj

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        try:
            other = self.check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        obj = type(self)(self)
        if isinstance(other, GeneralCurve):
            if not hasattr(obj, 'dy'):
                obj.dy = np.zeros_like(obj.y)
            # avoid modifying other by making a copy of it.
            other = GeneralCurve(other)
            if not hasattr(other, 'dy'):
                other.dy = np.zeros_like(other.dy)
            obj.dy = ((obj.y ** (other.y - 1) * other.y * obj.dy) **
                      2 + (np.log(obj.y) * obj.y ** other.y * other.dy) ** 2) ** 0.5
            obj.y = obj.y ** other.y
        elif isinstance(other, ErrorValue):
            if not hasattr(obj, 'dy'):
                obj.dy = np.zeros_like(obj.y)
            obj.dy = ((obj.y ** (other.val - 1) * other.val * obj.dy) **
                      2 + (np.log(obj.y) * obj.y ** other.val * other.err) ** 2) ** 0.5
            obj.y = obj.y ** other.val
        else:
            return NotImplemented
        return obj

    def __getitem__(self, name):
        if isinstance(name, numbers.Integral) or isinstance(name, slice) or isinstance(name, np.ndarray):
            d = dict()
            for k in self._controlled_attributes:
                d[k] = self.__getattribute__(k)[name]
            if isinstance(name, numbers.Integral):
                return d
            else:
                return type(self)(d)
        else:
            raise TypeError('indices must be integers, not %s' % type(name))

    def loglog(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt
        idx = np.isfinite(self.y) & np.isfinite(
            self.x) & (self.y > 0) & (self.x > 0)
        self._lastaxes = ax
        return ax.loglog(self.x[idx], self.y[idx], *args, **kwargs)

    def plot(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt
        self._lastaxes = ax
        return ax.plot(self.x, self.y, *args, **kwargs)

    def semilogx(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt
        idx = np.isfinite(self.y) & np.isfinite(self.x) & (self.x > 0)
        self._lastaxes = ax
        return ax.semilogx(self.x[idx], self.y[idx], *args, **kwargs)

    def semilogy(self, *args, **kwargs):
        idx = np.isfinite(self.y) & np.isfinite(self.x) & (self.y > 0)
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt
        self._lastaxes = ax
        return ax.semilogy(self.x[idx], self.y[idx], *args, **kwargs)

    def errorbar(self, *args, **kwargs):
        if hasattr(self, 'dx'):
            dx = self.dx
        else:
            dx = None
        if hasattr(self, 'dy'):
            dy = self.dy
        else:
            dy = None
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt
        self._lastaxes = ax
        return ax.errorbar(self.x, self.y, dy, dx, *args, **kwargs)

    def trim(self, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf):
        if xmin is None:
            xmin = -np.inf
        if xmax is None:
            xmax = np.inf
        if ymin is None:
            ymin = -np.inf
        if ymax is None:
            ymax = np.inf
        idx = (self.x <= xmax) & (self.x >= xmin) & (
            self.y >= ymin) & (self.y <= ymax)
        d = dict()
        for k in self._controlled_attributes:
            d[k] = self.__getattribute__(k)[idx]
        return type(self)(d)

    def trimzoomed(self):
        axis = self._lastaxes.axis()
        return self.trim(*axis)

    def sanitize(self, minval=-np.inf, maxval=np.inf, fieldname='y', discard_nonfinite=True):
        self_as_array = np.array(self)
        indices = (getattr(self, fieldname) > minval) & \
            (getattr(self, fieldname) < maxval)
        if discard_nonfinite:
            indices &= reduce(operator.and_, [np.isfinite(self_as_array[x])
                                              for x in self_as_array.dtype.names])
        self_as_array = self_as_array[indices]
        return type(self)(dict([(k, self_as_array[k]) for k in self_as_array.dtype.names]))

    def generate_errorbars(self):
        if not hasattr(self, 'dx'):
            self.dx = np.zeros_like(self.x)
        if not hasattr(self, 'dy'):
            self.dy = np.zeros_like(self.x)

    def save(self, filename, *args, **kwargs):
        self.generate_errorbars()
        self_as_array = np.array(self)
        headerline = '# ' + '\t'.join(*[self_as_array.dtype.names]) + '\n'
        filetobeclosed = False
        try:
            if hasattr(filename, 'write'):
                if 'b' in filename.mode:
                    filename.write(headerline.encode('utf-8'))
                else:
                    filename.write(headerline)
                fileopened = filename
                filetobeclosed = False
            elif isinstance(filename, str) and filename.upper().endswith('.GZ'):
                fileopened = gzip.GzipFile(filename, 'wb')
                fileopened.write(headerline.encode('utf-8'))
                filetobeclosed = True
            elif isinstance(filename, str):
                fileopened = open(filename, 'wb')
                fileopened.write(headerline.encode('utf-8'))
                filetobeclosed = True
            else:
                fileopened = filename
                filetobeclosed = False
            np.savetxt(fileopened, self_as_array, *args, **kwargs)
        finally:
            if filetobeclosed:
                fileopened.close()

    def interpolate(self, newx, **kwargs):
        if isinstance(newx, numbers.Number):
            if hasattr(self, 'dy'):
                return ErrorValue(np.interp(newx, self.x, self.y, **kwargs),
                                  np.interp(newx, self.x, self.dy, **kwargs))
            else:
                return np.interp(newx, self.x, self.y, **kwargs)
        d = {}
        for k in self.get_controlled_attributes():
            d[k] = np.interp(newx, self.x, getattr(self, k), **kwargs)
        return type(self)(d)

    @classmethod
    def merge(cls, first, last, xsep=None):
        if not (isinstance(first, cls) and isinstance(last, cls)):
            raise ValueError('Cannot merge types %s and %s together, only %s is supported.' % (
                type(first), type(last), cls))
        if xsep is not None:
            first = first.trim(xmax=xsep)
            last = last.trim(xmin=xsep)
        d = dict()
        for a in set(first.get_controlled_attributes()).intersection(set(last.get_controlled_attributes())):
            d[a] = np.concatenate((getattr(first, a), getattr(last, a)))
        return cls(d)

    def scalefactor(self, other, xmin=None, xmax=None, Npoints=None):
        """Calculate a scaling factor, by which this curve is to be multiplied to best fit the other one.

        Inputs:
            other: the other curve (an instance of GeneralCurve or of a subclass of it)
            xmin: lower cut-off (None to determine the common range automatically)
            xmax: upper cut-off (None to determine the common range automatically)
            Npoints: number of points to use in the common x-range (None defaults to the lowest value among
                the two datasets)

        Outputs:
            The scaling factor determined by interpolating both datasets to the same abscissa and calculating
                the ratio of their integrals, calculated by the trapezoid formula. Error propagation is
                taken into account.
        """
        if xmin is None:
            xmin = max(self.x.min(), other.x.min())
        if xmax is None:
            xmax = min(self.x.max(), other.x.max())
        data1 = self.trim(xmin, xmax)
        data2 = other.trim(xmin, xmax)
        if Npoints is None:
            Npoints = min(len(data1), len(data2))
        commonx = np.linspace(
            max(data1.x.min(), data2.x.min()), min(data2.x.max(), data1.x.max()), Npoints)
        I1 = data1.interpolate(commonx).momentum(1, True)
        I2 = data2.interpolate(commonx).momentum(1, True)
        return I2 / I1

    def unite(self, other, xmin=None, xmax=None, xsep=None,
              Npoints=None, scaleother=True, verbose=False, return_factor=False):
        if not isinstance(other, type(self)):
            raise ValueError(
                'Argument `other` should be an instance of class %s' % type(self))
        if scaleother:
            factor = other.scalefactor(self, xmin, xmax, Npoints)
            retval = type(self).merge(self, factor * other, xsep)
        else:
            factor = self.scalefactor(other, xmin, xmax, Npoints)
            retval = type(self).merge(factor * self, other, xsep)
        if verbose:
            print("Uniting two datasets.")
            print("   xmin   : ", xmin)
            print("   xmax   : ", xmax)
            print("   xsep   : ", xsep)
            print("   Npoints: ", Npoints)
            print("   Factor : ", factor)
        if return_factor:
            return retval, factor
        else:
            return retval

    def invert(self):
        """Calculate the inverse (i.e. swap x with y). No check is done if this
        makes sense! Other fields than x,y,dx,dy are lost from the inverted curve.
        """
        d = {}
        d['x'] = self.y
        d['y'] = self.x
        if hasattr(self, 'dx'):
            d['dy'] = self.dx
        if hasattr(self, 'dy'):
            d['dx'] = self.dy
        return type(self)(d)

    def momentum(self, exponent=1, errorrequested=True):
        """Calculate momenta (integral of y times x^exponent)
        The integration is done by the trapezoid formula (np.trapz).

        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned (true Gaussian
                error-propagation of the trapezoid formula)
        """
        y = self.y * self.x ** exponent
        m = np.trapz(y, self.x)
        if errorrequested:
            err = self.dy * self.x ** exponent
            dm = errtrapz(self.x, err)
            return ErrorValue(m, dm)
        else:
            return m

    def get_controlled_attributes(self):
        return ([x for x in list(self._special_names.values()) if hasattr(self, x)] +
                [x for x in self._controlled_attributes
                 if x not in list(self._special_names.keys()) and
                 x not in list(self._special_names.values())])

    def __array__(self, dtype=None, attrs=None):
        """Make a structured numpy array from the current dataset.
        """
        if attrs == None:
            attrs = self.get_controlled_attributes()
        values = [getattr(self, k) for k in attrs]
        if dtype is None:
            return np.array(list(zip(*values)), dtype=list(zip(attrs, [v.dtype for v in values])))
        else:
            return np.array(list(zip(*values)), dtype=list(zip(attrs, [dtype for v in values])))

    def sorted(self, order='x'):
        """Sort the current dataset according to 'order' (defaults to '_x').
        """
        a = np.array(self)
        self_as_array = np.sort(a, order=order)
        for k in self_as_array.dtype.names:
            setattr(self, k, self_as_array[k])
        return self

    def odrfit(self, function, parameters_init, xname='x', yname='y', dyname='dy', dxname='dx', **kwargs):
        """Orthogonal distance regression to the dataset.

        Inputs:
        -------
            `function`: a callable, corresponding to the function to be fitted.
                Should have the following signature::

                    >>> function(x, par1, par2, par3, ...)

                where ``par1``, ``par2``, etc. are the values of the parameters
                to be fitted.

            `parameters_init`: a sequence (tuple or list) of the initial values
                for the parameters to be fitted. Their ordering should be the
                same as that of the arguments of `function`

            other keyword arguments are given to `nonlinear_odr()`
                without any modification.

        Outputs:
        --------
            `par1_fit`, `par2_fit`, etc.: the best fitting values of the parameters.
            `statdict`: a dictionary with various status data, such as `R2`, `DoF`,
                `Chi2_reduced`, `Covariance`, `Correlation_coeffs` etc. For a full
                list, see the help of `sastool.misc.easylsq.odr_fit()`
            `func_value`: the value of the function at the best fitting parameters,
                represented as an instance of the same class as this curve.

        Notes:
        ------
            The fitting itself is done via sastool.misc.easylsq.nonlinear_odr()

        """
        obj = self.sanitize(fieldname=yname).sanitize(fieldname=xname)
        if not hasattr(obj, dyname):
            dy = None
        else:
            dy = getattr(obj, dyname)
        if not hasattr(obj, dxname):
            dx = None
        else:
            dx = getattr(obj, dxname)
        ret = nonlinear_odr(getattr(obj, xname), getattr(
            obj, yname), dx, dy, function, parameters_init, **kwargs)
        funcvalue = type(self)(getattr(obj, xname), ret[-1]['func_value'])
        return ret + (funcvalue,)

    def fit(self, function, parameters_init, xname='x', yname='y', dyname='dy', **kwargs):
        """Non-linear least-squares fit to the dataset.

        Inputs:
        -------
            `function`: a callable, corresponding to the function to be fitted.
                Should have the following signature::

                    >>> function(x, par1, par2, par3, ...)

                where ``par1``, ``par2``, etc. are the values of the parameters
                to be fitted.

            `parameters_init`: a sequence (tuple or list) of the initial values
                for the parameters to be fitted. Their ordering should be the
                same as that of the arguments of `function`

            other keyword arguments are given to `nonlinear_leastsquares()`
                without any modification.

        Outputs:
        --------
            `par1_fit`, `par2_fit`, etc.: the best fitting values of the parameters.
            `statdict`: a dictionary with various status data, such as `R2`, `DoF`,
                `Chi2_reduced`, `Covariance`, `Correlation_coeffs` etc. For a full
                list, see the help of `sastool.misc.easylsq.nlsq_fit()`
            `func_value`: the value of the function at the best fitting parameters,
                represented as an instance of the same class as this curve.

        Notes:
        ------
            The fitting itself is done via sastool.misc.easylsq.nonlinear_leastsquares()
        """
        obj = self.sanitize(fieldname=yname).sanitize(fieldname=xname)
        if not hasattr(obj, dyname):
            dy = np.ones(len(obj), np.double) * np.nan
        else:
            dy = getattr(obj, dyname)
        ret = nonlinear_leastsquares(getattr(obj, xname), getattr(
            obj, yname), dy, function, parameters_init, **kwargs)
        funcvalue = type(self)(getattr(obj, xname), ret[-1]['func_value'])
        return ret + tuple([funcvalue])

    @classmethod
    def simultaneous_fit(cls, list_of_curves, function, params_init, xname='x', yname='y', dyname='dy', **kwargs):
        """Non-linear simultaneous least-squares fit to several curves.

        Inputs:
        -------
            `list_of_curves`: a sequence of instances of this class for fitting.
            `function`: a callable, corresponding to the function to be fitted.
                Should have the following signature::

                    >>> function(x, par1, par2, par3, ...)

                where ``par1``, ``par2``, etc. are the values of the parameters
                to be fitted.

            `params_init`: a sequence of sequences of the initial values
                for the parameters to be fitted. Should be laid out as:
                ((par1_dataset1, par2_dataset1, par3_dataset1, ...),
                 (par1_dataset2, par2_dataset2, par3_dataset2, ...),
                 ...)
                Each parameter can be:
                    . a float
                    . an instance of sastool.misc.ErrorValue
                    . None (means that this parameter is to be held the same
                        as the corresponding parameter before this; of course
                        no None-s can occur among the parameters for the first
                        dataset)
                    . an instance of sastool.misc.easylsq.FixedParameter (in this
                        case the parameter will be held fixed and not be fitted.
                        This is useful if you want the common fitting function to
                        behave *slightly* differently for each dataset).

            other keyword arguments are given to `nonlinear_leastsquares()`
                without any modification.

        Outputs:
        --------
            a list of fitting results for each curve. Each element of the list
            is a tuple and has the format as returned by GeneralCurve.fit():

                (par1, par2, par3, ... , statdict, fittedcurve)

        """
        if not all(isinstance(x, cls) for x in list_of_curves):
            raise ValueError('All curves should be an instance of ' +
                             cls + ', not ' + ', '.join([type(x) for x in list_of_curves]))

        def getdy(curve):
            if hasattr(curve, dyname):
                return getattr(curve, dyname)
            else:
                return np.ones(len(curve), np.double) * np.nan
        list_of_curves = [a.sanitize(fieldname=yname).sanitize(
            fieldname=xname) for a in list_of_curves]
        ret = simultaneous_nonlinear_leastsquares(tuple([getattr(a, xname) for a in list_of_curves]),
                                                  tuple([getattr(a, yname)
                                                         for a in list_of_curves]),
                                                  tuple(
                                                      [getdy(a) for a in list_of_curves]),
                                                  function, params_init, **kwargs)
        results = []
        for pars, idx in zip(ret[:-1], itertools.count(0)):
            statdict = ret[-1].copy()
            for name in ['R2', 'Chi2', 'Chi2_reduced', 'DoF', 'Covariance', 'Correlation_coeffs', 'func_value']:
                statdict[name] = statdict[name][idx]
            funcval = cls(
                getattr(list_of_curves[idx], xname), statdict['func_value'])
            results.append(tuple(pars) + (statdict, funcval))
        return results

    @classmethod
    def average(cls, *args):
        args = list(args)
        if len(args) == 1:
            return args[0]
        for i in range(1, len(args)):
            args[i] = args[0].check_arithmetic_compatibility(args[i])
        d = {}
        if not all([a.has_valid_dx() for a in args]):
            d['x'] = np.mean([a.x for a in args], 0)
            d['dx'] = np.std([a.x for a in args], 0)
        else:
            sumweight = np.sum([1.0 / a.dx ** 2 for a in args], 0)
            d['x'] = np.sum([a.x / a.dx ** 2 for a in args], 0) / sumweight
            d['dx'] = 1.0 / sumweight**0.5
        if not all([a.has_valid_dy() for a in args]):
            d['y'] = np.mean([a.y for a in args], 0)
            d['dy'] = np.std([a.y for a in args], 0)
        else:
            sumweight = np.sum([1.0 / a.dy ** 2 for a in args], 0)
            d['y'] = np.sum([a.y / a.dy ** 2 for a in args], 0) / sumweight
            d['dy'] = 1.0 / sumweight**0.5
        return cls(d)

    @classmethod
    def average_masked(cls, *args):
        args = list(args)
        if len(args) == 1:
            return args[0]
        for i in range(1, len(args)):
            if args[i].shape != args[0].shape:
                raise ValueError('Incompatible shape!')
        datalen = args[0].size
        datanum = len(args)
        mask = np.array([x.mask for x in args], dtype=np.bool)
        d = {}
        d['x'] = np.ma.array([x.x for x in args], mask=mask)
        d['y'] = np.ma.array([x.y for x in args], mask=mask)
        # if there are invalid or missing x error bars
        if all([a.has_valid_dx() for a in args]):
            d['dx'] = np.ma.array([x.dx for x in args], mask=mask)
            np.ma.array
            # d['x']=
        # if there are invalid or missing x error bars
        if all([a.has_valid_dy() for a in args]):
            d['dy'] = np.array([x.dy for x in args])

    def has_valid_dx(self):
        return hasattr(self, 'dx') and np.absolute(self.dx).sum() > 0

    def has_valid_dy(self):
        return hasattr(self, 'dy') and np.absolute(self.dy).sum() > 0


class SASCurve(GeneralCurve):
    _default_special_names = [
        ('x', 'q'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'qError')]


class SASPixelCurve(GeneralCurve):
    _default_special_names = [
        ('x', 'pixel'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'pixelError')]


class SASAzimuthalCurve(GeneralCurve):
    _default_special_names = [
        ('x', 'phi'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'phiError')]
