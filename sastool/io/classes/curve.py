'''
Created on Jul 5, 2012

@author: andris
'''

import numpy as np
import collections
import numbers
import matplotlib.pyplot as plt
import gzip
import itertools
import operator
import functools
import warnings

from ...dataset import ArithmeticBase, ErrorValue
from ...fitting.easylsq import nlsq_fit

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
    def __init__(self, value = None, name = None, obj = None):
        if isinstance(value, type(self)):
            self.value = value.value
            self.name = value.name
        else:
            self.name = name
            if not isinstance(value, np.ndarray):
                raise TypeError("Cannot instantiate a ControlledVectorAttribute with a type %s" % type(value))
            self.value = obj.check_compatibility(value, self.name)
    def __get__(self, obj, type = None):
        return self.value
    def __set__(self, obj, value):
        self.value = obj.check_compatibility(value, self.name)
    def __delete__(self, obj):
        if hasattr(obj, '_controlled_attributes'):
            obj._controlled_attributes.remove(self.name)

class GeneralCurve(ArithmeticBase):
    _xtol = 1e-3
    _default_special_names = [('x', 'X'), ('y', 'Y'), ('dy', 'DY'), ('dx', 'DX')]
    def __init__(self, *args, **kwargs):
        kwargs = self._initialize_kwargs_for_readers(kwargs)
        self._controlled_attributes = []
        self._special_names = kwargs['special_names']
        if len(args) == 0:
            #empty constructor
            self._special_names = kwargs['special_names']
            self.x = None
            self.y = None
            self.dy = None
            self.dx = None
        elif isinstance(args[0], type(self)):
            #copy constructor
            self._special_names = args[0]._special_names
            for name in args[0]._controlled_attributes:
                self.add_dataset(name, getattr(args[0], name))
        elif isinstance(args[0], dict):
            for k in args[0]:
                self.add_dataset(k, args[0][k])
        elif isinstance(args[0], basestring):
            kwargs_for_loadtxt = kwargs.copy()
            del kwargs_for_loadtxt['special_names']
            columnnames = itertools.chain(self._special_names.values(),
                                        itertools.imap(lambda x:'column%d' % x,
                                                       itertools.count(len(self._special_names))
                                                       )
                                        )
            with open(args[0], 'rt') as f:
                firstline = f.readline()
                if firstline.startswith('#'):
                    columnnames = firstline[1:].strip().split()

            m = np.loadtxt(args[0], **kwargs_for_loadtxt)
            for i, cn in itertools.izip(range(m.shape[1]), columnnames):
                self.add_dataset(cn, m[:, i])
        else:
            for a, name in zip(args, self._special_names.values()):
                self.add_dataset(name, a)
        for a, name in [(kwargs[k], k) for k in kwargs if isinstance(kwargs[k], np.ndarray) or isinstance(kwargs[k], collections.Sequence)]:
            self.add_dataset(name, a)
    def __getattribute__(self, name):
        #hack to allow instance descriptors. http://blog.brianbeck.com/post/74086029/instance-descriptors
        value = object.__getattribute__(self, name)
        if isinstance(value, ControlledVectorAttribute):
            value = value.__get__(self, self.__class__)
        return value
    def __setattr__(self, name, value):
        #hack to allow instance descriptors. http://blog.brianbeck.com/post/74086029/instance-descriptors
        if hasattr(self, '_special_names') and name in self._special_names.values():
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
            kwargs['special_names'] = collections.OrderedDict(self._default_special_names)
        return kwargs
    @property
    def xname(self):
        """The name of the dataset treated as the abscissa."""
        return self.get_specialname('x')
    @xname.setter
    def xname(self, value):
        return self.set_specialname('x', value)

    @property
    def yname(self):
        """The name of the dataset treated as the ordinate."""
        return self.get_specialname('y')
    @yname.setter
    def yname(self, value):
        return self.set_specialname('y', value)

    @property
    def dxname(self):
        """The name of the dataset treated as the error of the abscissa."""
        return self.get_specialname('dx')
    @dxname.setter
    def dxname(self, value):
        return self.set_specialname('dx', value)

    @property
    def dyname(self):
        """The name of the dataset treated as the error of the ordinate."""
        return self.get_specialname('dy')
    @dyname.setter
    def dyname(self, value):
        return self.set_specialname('dy', value)

    @property
    def x(self):
        return self.__getattribute__(self.xname)
    @x.setter
    def x(self, value):
        return self.add_dataset(self.xname, value)
    @property
    def y(self):
        return self.__getattribute__(self.yname)
    @y.setter
    def y(self, value):
        return self.add_dataset(self.yname, value)
    @property
    def dx(self):
        return self.__getattribute__(self.dxname)
    @dx.setter
    def dx(self, value):
        return self.add_dataset(self.dxname, value)
    @property
    def dy(self):
        return self.__getattribute__(self.dyname)
    @dy.setter
    def dy(self, value):
        return self.add_dataset(self.dyname, value)
    def set_specialname(self, specname, newname):
        if newname == specname:
            raise ValueError('Name `%s` not supported for %s. Try `%s`.' % (newname, specname, specname.upper()))
        self._special_names[specname] = newname
    def get_specialname(self, specname):
        return self._special_names[specname]
    def check_compatibility(self, value, name):
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
        if value is None:
            return self.remove_dataset(name)
        retval = object.__setattr__(self, name, ControlledVectorAttribute(value, name, self))
        self._controlled_attributes.append(name)
        return retval
    def remove_dataset(self, name):
        if hasattr(self, name):
            self.__delattr__(name)
        else:
            print "Skipping removing ", name
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
            if len(other) == len(self):
                return ErrorValue(other.flatten())
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, type(self)):
            if not len(other) == len(self):
                raise ValueError('Incompatible shape!')
            if hasattr(other, 'dx') and hasattr(self, 'dx') and ((self.dx ** 2 + other.dx ** 2).sum() > 0):
                if (((self.dx ** 2 + other.dx ** 2) ** 0.5 - np.absolute(self.x - other.x)) <= 0).all():
                    return other
                else:
                    raise ValueError('Abscissae are not the same within error!')
            elif np.absolute(other.x - self.x).max() < max(self._xtol, other._xtol):
                return other
            else:
                raise ValueError('Incompatible abscissae!')
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
        elif isinstance(other, type(self)):
            self.x = 0.5 * (self.x + other.x)
            self.y = self.y + other.y
            if not hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = other.dx
            elif hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = 0.5 * (self.dx + other.dx)
            #the other two cases do not need explicit treatment.
            if not hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = other.dy
            elif hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = np.sqrt(self.dy ** 2 + other.dy ** 2)
        else:
            return NotImplemented
        return self
    def __imul__(self, other):
        try:
            other = self.check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        if isinstance(other, ErrorValue):
            if not hasattr(self, 'dy'):
                self.dy = np.zeros_like(self.y)
            self.dy = np.sqrt(self.dy ** 2 * other.val ** 2 + other.err ** 2 * self.y ** 2)
            self.y = self.y * other.val
        elif isinstance(other, type(self)):
            if not hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = other.dx
            elif hasattr(self, 'dx') and hasattr(other, 'dx'):
                self.dx = 0.5 * (self.dx + other.dx)
            #the other two cases do not need explicit treatment.
            if not hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = other.dy
            elif hasattr(self, 'dy') and hasattr(other, 'dy'):
                self.dy = np.sqrt(self.dy ** 2 * other.y ** 2 + other.dy ** 2 * self.y ** 2)
            self.x = 0.5 * (self.x + other.x)
            self.y = self.y * other.y
        else:
            return NotImplemented
        return self
    def __neg__(self):
        obj = type(self)(self)  # make a copy
        obj.y = -obj.y
        return obj
    def _recip(self):
        obj = type(self)(self)
        if hasattr(self, 'dy'):
            obj.dy = (1.0 * obj.dy) / (obj.y * obj.y)
        obj.y = 1.0 / obj.y
        return obj
    def __getitem__(self, name):
        if isinstance(name, numbers.Integral) or isinstance(name, slice):
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
        return plt.loglog(self.x, self.y, *args, **kwargs)
    def plot(self, *args, **kwargs):
        return plt.plot(self.x, self.y, *args, **kwargs)
    def semilogx(self, *args, **kwargs):
        return plt.semilogx(self.x, self.y, *args, **kwargs)
    def semilogy(self, *args, **kwargs):
        return plt.semilogy(self.x, self.y, *args, **kwargs)
    def errorbar(self, *args, **kwargs):
        if hasattr(self, 'dx'):
            dx = self.dx
        else:
            dx = None
        if hasattr(self, 'dy'):
            dy = self.dy
        else:
            dy = None
        return plt.errorbar(self.x, self.y, dy, dx, *args, **kwargs)
    def trim(self, xmin = -np.inf, xmax = np.inf, ymin = -np.inf, ymax = np.inf):
        idx = (self.x <= xmax) & (self.x >= xmin) & (self.y >= ymin) & (self.y <= ymax)
        d = dict()
        for k in self._controlled_attributes:
            d[k] = self.__getattribute__(k)[idx]
        return type(self)(d)
    def trimzoomed(self):
        axis = plt.axis()
        return self.trim(*axis)
    def sanitize(self, minval = -np.inf, maxval = np.inf, fieldname = 'y', discard_nonfinite = True):
        self_as_array = np.array(self)
        indices = (getattr(self, fieldname) >= minval) & \
            (getattr(self, fieldname) <= maxval)
        if discard_nonfinite:
            indices &= reduce(operator.and_, [np.isfinite(self_as_array[x]) \
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
        if hasattr(filename, 'write'):
            filename.write(headerline)
            fileopened = filename
            filetobeclosed = False
        elif isinstance(filename, basestring) and filename.upper().endswith('.GZ'):
            fileopened = gzip.GzipFile(filename, 'wb')
            fileopened.write(headerline)
            filetobeclosed = True
        elif isinstance(filename, basestring):
            fileopened = open(filename, 'wb')
            fileopened.write(headerline)
            filetobeclosed = True
        else:
            fileopened = filename
            filetobeclosed = False
        np.savetxt(fileopened, self_as_array, *args, **kwargs)
        if filetobeclosed:
            fileopened.close()
    def interpolate(self, newx, **kwargs):
        d = {}
        for k in self.get_controlled_attributes():
            d[k] = np.interp(newx, self.x, getattr(self, k), **kwargs)
        return type(self)(d)
    @classmethod
    def merge(cls, first, last, xsep = None):
        if not (isinstance(first, cls) and isinstance(last, cls)):
            raise ValueError('Cannot merge types %s and %s together, only %s is supported.' % (type(first), type(last), cls))
        if xsep is not None:
            first = first.trim(xmax = xsep)
            last = last.trim(xmin = xsep)
        d = dict()
        for a in set(first.get_controlled_attributes()).intersection(set(last.get_controlled_attributes())):
            d[a] = np.concatenate((getattr(first, a), getattr(last, a)))
        return cls(d)
    def unite(self, other, xmin = None, xmax = None, xsep = None,
              Npoints = None, scaleother = True, verbose = False, return_factor=False):
        if not isinstance(other, type(self)):
            raise ValueError('Argument `other` should be an instance of class %s' % type(self))
        if xmin is None:
            xmin = max(self.x.min(), other.x.min())
        if xmax is None:
            xmax = min(self.x.max(), other.x.max())
        data1 = self.trim(xmin, xmax)
        data2 = other.trim(xmin, xmax)
        if Npoints is None:
            Npoints = min(len(data1), len(data2))
        commonx=np.linspace(max(data1.x.min(),data2.x.min()),min(data2.x.max(),data1.x.max()),Npoints)
        I1 = data1.interpolate(commonx).momentum(1, True)
        I2 = data2.interpolate(commonx).momentum(1, True)
        if scaleother:
            factor=I1/I2
            retval=type(self).merge(self, factor * other, xsep)
        else:
            factor=I2/I1
            retval=type(self).merge(factor * self, other, xsep)
        if verbose:
            print "Uniting two datasets."
            print "   xmin   : ", xmin
            print "   xmax   : ", xmax
            print "   xsep   : ", xsep
            print "   I_1    : ", I1
            print "   I_2    : ", I2
            print "   Npoints: ", Npoints
            print "   Factor : ", I1 / I2
        if return_factor:
            return retval, factor
        else:
            return retval
    def momentum(self, exponent = 1, errorrequested = True):
        """Calculate momenta (integral of y times x^exponent)
        The integration is done by the trapezoid formula (np.trapz).
        
        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned (true Gaussian
                error-propagation of the trapezoid formula)
        """
        y = self.x * self.y ** exponent
        m = np.trapz(y, self.x)
        if errorrequested:
            err = self.dy * self.x ** exponent
            dm = errtrapz(self.x, err)
            return ErrorValue(m, dm)
        else:
            return m
    def get_controlled_attributes(self):
        return ([x for x in self._special_names.values() if hasattr(self, x)] +
            [x for x in self._controlled_attributes \
             if x not in self._special_names.keys() and \
             x not in self._special_names.values()])
    def __array__(self, attrs = None):
        """Make a structured numpy array from the current dataset.
        """
        if attrs == None:
            attrs = self.get_controlled_attributes()
        values = [getattr(self, k) for k in attrs]
        return np.array(zip(*values), dtype = zip(attrs, [v.dtype for v in values]))
    def sorted(self, order = 'x'):
        """Sort the current dataset according to 'order' (defaults to '_x').
        """
        a = np.array(self)
        self_as_array = np.sort(a, order = order)
        for k in self_as_array.dtype.names:
            setattr(self, k, self_as_array[k])
        return self
    def fit(self, function, parameters_init, **kwargs):
        pars, dpars, statdict = nlsq_fit(self.x, self.y, self.dy, function, parameters_init, **kwargs)
        return [ErrorValue(p, dp) for p, dp in zip(pars, dpars)], statdict
    @classmethod
    def average(cls, *args):
        args=list(args)
        if len(args)==1:
            return args[0]
        for i in range(1,len(args)):
            args[i]=args[0].check_arithmetic_compatibility(args[i])
        d={}
        if not all([a.has_valid_dx() for a in args]):
            d['x']=np.mean([a.x for a in args],0)
            d['dx']=np.std([a.x for a in args],0)
        else:
            sumweight=np.sum([1.0/a.dx**2 for a in args],0)
            d['x']=np.sum([a.x/a.dx**2 for a in args],0)/sumweight
            d['dx']=1.0/sumweight
        if not all([a.has_valid_dy() for a in args]):
            d['y']=np.mean([a.y for a in args],0)
            d['dy']=np.std([a.y for a in args],0)
        else:
            sumweight=np.sum([1.0/a.dy**2 for a in args],0)
            d['y']=np.sum([a.y/a.dy**2 for a in args],0)/sumweight
            d['dy']=1.0/sumweight
        return cls(d)
    def has_valid_dx(self):
        return hasattr(self,'dx') and np.absolute(self.dx).sum()>0
    def has_valid_dy(self):
        return hasattr(self,'dy') and np.absolute(self.dy).sum()>0
    
class SASCurve(GeneralCurve):
    _default_special_names = [('x', 'q'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'qError')]

class SASPixelCurve(GeneralCurve):
    _default_special_names = [('x', 'pixel'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'pixelError')]

class SASAzimuthalCurve(GeneralCurve):
    _default_special_names = [('x', 'phi'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'phiError')]