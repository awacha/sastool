'''
Created on Jul 5, 2012

@author: andris
'''

import numpy as np
import collections
import numbers
import matplotlib.pyplot as plt

from .. import onedim
from ...dataset import ArithmeticBase, ErrorValue

class ControlledVectorAttribute(object):
    def __init__(self, value = None, name = None, obj = None):
        self.name = name
        self.value = obj.check_compatibility(value, self.name)
    def __get__(self, obj, type = None):
        return self.value
    def __set__(self, obj, value):
        self.value = obj.check_compatibility(value, self.name)
    def __delete__(self, obj):
        print "CVA.__delete__, name:", self.name
        if hasattr(obj, '_controlled_attributes'):
            obj._controlled_attributes.remove(self.name)

class SASCurve(ArithmeticBase):
    _xtol = 1e-3
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
        elif isinstance(args[0], SASCurve):
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
            m = np.loadtxt(args[0], **kwargs_for_loadtxt)
            for i in range(m.shape[1]):
                self.add_dataset(self._special_names.values()[i], m[:, i])
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
    @staticmethod
    def _initialize_kwargs_for_readers(kwargs):
        if 'special_names' not in kwargs:
            kwargs['special_names'] = collections.OrderedDict([('x', 'q'), ('y', 'Intensity'), ('dy', 'Error'), ('dx', 'qError')])
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
        object.__setattr__(self, name, ControlledVectorAttribute(value, name, self))
        self._controlled_attributes.append(name)
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
        if isinstance(other, numbers.Number):
            return ErrorValue(other)
        elif isinstance(other, np.ndarray):
            if len(other) == len(self):
                return ErrorValue(other.flatten())
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, SASCurve):
            if not len(other) == len(self):
                raise ValueError('Incompatible shape!')
            if hasattr(other, 'dx') and hasattr(self, 'dx') and (self.dx ** 2 + other.dy ** 2).sum() > 0:
                if (((self.dx ** 2 + other.dx ** 2) - np.absolute(self.x - other.x)) > 0).all():
                    return other
                else:
                    raise ValueError('Abscissae are not the same within error!')
            elif np.absolute(other.x - self.x).max() < max(self._xtol, other._xtol):
                return other
            else:
                raise ValueError('Incompatible abscissa!')
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
        elif isinstance(other, SASCurve):
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
        elif isinstance(other, SASCurve):
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
    def sanitize(self):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError
    def interpolate(self):
        raise NotImplementedError
    def merge(self):
        raise NotImplementedError
    def unite(self):
        raise NotImplementedError
