# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:18:05 2011

@author: -
"""
from arithmetic import ArithmeticBase
import numpy as np
import numbers
import math
import collections

class ErrorValue(ArithmeticBase):
    """Class to hold a value and its absolute error. Basic arithmetic
    operations are supported.
    """
    def __init__(self, val, err=None):
        ArithmeticBase.__init__(self)
        if isinstance(val, numbers.Number):
            self.val = float(val)
            if isinstance(err, numbers.Number):
                self.err = float(err)
            elif err is None:
                self.err = 0.0
        elif isinstance(val, np.ndarray):
            self.val = val.copy()
            if isinstance(err, np.ndarray) and (err.shape == val.shape):
                self.err = err.copy()
            elif err is None:
                self.err = np.zeros_like(self.val)
            else:
                raise ValueError('Incompatible shape of value and its error!')
        elif isinstance(val, ErrorValue):
            if isinstance(val.val, np.ndarray):
                self.val = val.val.copy()
                self.err = val.err.copy()
            elif isinstance(val.val, numbers.Number):
                self.val = val.val
                self.err = val.err
        elif isinstance(val, collections.Sequence):
            if all(isinstance(v, ErrorValue) for v in val):
                self.val = np.array([v.val for v in val])
                self.err = np.array([v.err for v in val])
            elif all(isinstance(v, numbers.Number) for v in val):
                self.val = np.array(val)
                self.err = np.zeros_like(self.val)
            else:
                raise ValueError('If instantiated with a sequence, all elements of it must either be ErrorValues or numbers.')
        else:
            raise ValueError('ErrorValue class can hold only Python numbers or numpy ndarrays, got %s!' % type(val))
    def copy(self):
        """Make a deep copy of this instance"""
        return ErrorValue(self.val, self.err)
    def __neg__(self):
        return ErrorValue(-self.val, self.err)
    def _recip(self):
        """Calculate the reciprocal of this instance"""
        return ErrorValue(1.0 / self.val, self.err / (self.val * self.val))
    def __iadd__(self, value):
        try:
            value = ErrorValue(value)
        except ValueError:
            return NotImplemented
        self.val = self.val + value.val
        self.err = np.sqrt(self.err ** 2 + value.err ** 2)
        return self
    def __imul__(self, value):
        #print "Errorvalue.__imul__"
        try:
            value = ErrorValue(value)
        except ValueError:
            #print "Errorvalue.__imul__ not implemented."
            return NotImplemented
        self.err = np.sqrt(self.err * self.err * value.val * value.val +
                             value.err * value.err * self.val * self.val)
        self.val = self.val * value.val
        return self
    def __str__(self):
        return self.tostring()
    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        try:
            other = ErrorValue(other)
        except ValueError:
            return NotImplemented
        err = ((self.val ** (other.val - 1) * other.val * self.err) ** 2 + (np.log(self.val) * self.val ** other.val * other.err) ** 2) ** 0.5
        val = self.val ** other.val
        return ErrorValue(val, err)
    def __repr__(self):
        return 'ErrorValue(' + repr(self.val) + ' +/- ' + repr(self.err) + ')'
    def __float__(self):
        return float(self.val)
    def __trunc__(self):
        return long(self.val)
    def __array__(self, dt=None):
        if dt is None:
            return np.array(self.val)
        else:
            return np.array(self.val, dt)
    def tostring(self, extra_digits=0):
        if isinstance(self.val, numbers.Real):
            try:
                Ndigits = -int(math.floor(math.log10(self.err))) + extra_digits
            except (OverflowError, ValueError):
                return str(self.val) + ' +/- ' + str(self.err)
            else:
                return str(round(self.val, Ndigits)) + ' +/- ' + str(round(self.err, Ndigits))
        return str(self.val) + ' +/- ' + str(self.err)
    def sin(self):
        return ErrorValue(np.sin(self.val), np.abs(np.cos(self.val) * self.err))
    def cos(self):
        return ErrorValue(np.cos(self.val), np.abs(np.sin(self.val) * self.err))
    def tan(self):
        return ErrorValue(np.tan(self.val), np.abs(1 + np.tan(self.val) ** 2) * self.err)
    def sqrt(self):
        return self ** 0.5
    def sinh(self):
        return ErrorValue(np.sinh(self.val), np.abs(np.cosh(self.val) * self.err))
    def cosh(self):
        return ErrorValue(np.cosh(self.val), np.abs(np.sinh(self.val) * self.err))
    def tanh(self):
        return ErrorValue(np.tanh(self.val), np.abs(1 - np.tanh(self.val) ** 2) * self.err)
    def arcsin(self):
        return ErrorValue(np.arcsin(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))
    def arccos(self):
        return ErrorValue(np.arccos(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))
    def arcsinh(self):
        return ErrorValue(np.arcsinh(self.val), np.abs(self.err / np.sqrt(1 + self.val ** 2)))
    def arccosh(self):
        return ErrorValue(np.arccosh(self.val), np.abs(self.err / np.sqrt(self.val ** 2 - 1)))
    def arctanh(self):
        return ErrorValue(np.arctanh(self.val), np.abs(self.err / (1 - self.val ** 2)))
    def arctan(self):
        return ErrorValue(np.arctan(self.val), np.abs(self.err / (1 + self.val ** 2)))
    def log(self):
        return ErrorValue(np.log(self.val), np.abs(self.err / self.val))
    def exp(self):
        return ErrorValue(np.exp(self.val), np.abs(self.err * np.exp(self.val)))
