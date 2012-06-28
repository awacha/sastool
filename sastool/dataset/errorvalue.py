# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:18:05 2011

@author: -
"""
from arithmetic import ArithmeticBase
import numpy as np
import numbers

class ErrorValue(ArithmeticBase):
    """Class to hold a value and its absolute error. Basic arithmetic
    operations are supported.
    """
    def __init__(self, val, err = None):
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
        else:
            raise ValueError('ErrorValue class can hold only Python numbers or numpy ndarrays!')
    def copy(self):
        """Make a deep copy of this instance"""
        return ErrorValue(self.val, self.err)
    def __neg__(self):
        return ErrorValue(-self.val, self.err)
    def _recip(self):
        """Calculate the reciprocal of this instance"""
        return ErrorValue(1.0 / self.val, self.err / (self.val * self.val))
    def __iadd__(self, value):
        value = ErrorValue(value)
        self.val = self.val + value.val
        self.err = np.sqrt(self.err ** 2 + value.err ** 2)
        return self
    def __imul__(self, value):
        value = ErrorValue(value)
        self.err = np.sqrt(self.err * self.err * value.val * value.val +
                             value.err * value.err * self.val * self.val)
        self.val = self.val * value.val
        return self
    def __str__(self):
        return str(self.val) + ' +/- ' + str(self.err)
    def __unicode__(self):
        return unicode(self.val) + ' +/- ' + unicode(self.err)
    def __repr__(self):
        return 'ErrorValue(' + repr(self.val) + ' +/- ' + repr(self.err) + ')'
