# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:18:05 2011

@author: -
"""
from arithmetic import ArithmeticBase
import numpy as np

class ErrorValue(ArithmeticBase):
    """Class to hold a scalar value and its absolute error. Basic arithmetic
    operations are supported.
    """
    def __init__(self, val, err = 0):
        ArithmeticBase.__init__(self)
        self.val = val
        self.err = err
    def copy(self):
        """Make a deep copy of this instance"""
        return ErrorValue(self.val, self.err)
    def __neg__(self):
        obj = self.copy()
        obj.val = -obj.val
        return obj
    def _recip(self):
        """Calculate the reciprocal of this instance"""
        obj = self.copy()
        obj.err = self.err/(self.val*self.val)
        obj.val = 1.0/self.val
        return obj
    def __iadd__(self, value):
        if isinstance(value, ErrorValue):
            self.val += value.val
            self.err = np.sqrt(self.err*self.err+value.err*value.err)
        else:
            self.val += value
        return self
    def __imul__(self, value):
        if isinstance(value, ErrorValue):
            self.err = np.sqrt(self.err*self.err*value.val*value.val+
                             value.err*value.err*self.val*self.val)
            self.val *= value.val
        else:
            self.val *= value
            self.err *= value
        return self
    def __str__(self):
        return "%g +/- %g" % (self.val, self.err)
