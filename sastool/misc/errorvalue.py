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
import re

__all__ = ['ErrorValue']

class ErrorValue(ArithmeticBase):
    """Class to hold a value and its uncertainty (1sigma, absolute error etc.).

    Main features:
        o Easy access to the value and its uncertainty through the `.val` and
            `.err` fields.
        o Basic arithmetic operations (+, -, *, /, **) are supported.
        o Can be instantiated easily by ``ErrorValue(value, [error])`` from:
            - scalar numbers
            - numpy ndarrays
            - other instances of `ErrorValue` (aka. "copy constructor")
            - homogeneous Python sequences of `ErrorValue` instances or scalar
                numbers
        o Intuitive string representation; the number of decimals is determined
            by the magnitude of the uncertainty (see `.tostring()` method)
        o Drop-in usage instead of `float` and `int` and `np.ndarray` by the
            conversion methods.
        o Basic trigonometric and hyperbolic functions are supported as methods,
            e.g. ``ev.sin()``, etc.
        o Sampling of random numbers from the Gaussian distribution described
            by the value and uncertainty fields by a single call to `.random()`
        o Evaluating complicated functions by `ErrorValue.evalfunc()`; error
            propagation is done by a Monte Carlo approach.
        o List-like indexing and slicing if ``val`` and ``err`` are arrays. Only
            read access is supported.
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
        return self.__class__(self.val, self.err)
    def __neg__(self):
        return self.__class__(-self.val, self.err)
    def _recip(self):
        """Calculate the reciprocal of this instance"""
        return self.__class__(1.0 / self.val, self.err / (self.val * self.val))
    def __getitem__(self, key):
        return self.__class__(self.val[key], self.err[key])
    def __iadd__(self, value):
        try:
            value = ErrorValue(value)
        except ValueError:
            return NotImplemented
        self.val = self.val + value.val
        self.err = np.sqrt(self.err ** 2 + value.err ** 2)
        return self
    def __imul__(self, value):
        try:
            value = ErrorValue(value)
        except ValueError:
            return NotImplemented
        self.err = np.sqrt(self.err * self.err * value.val * value.val + 
                             value.err * value.err * self.val * self.val)
        self.val = self.val * value.val
        return self
    def __str__(self):
        return self.tostring(plusminus=' +/- ')
    def __unicode__(self):
        return self.tostring(plusminus=u' \xb1 ')
    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        try:
            other = ErrorValue(other)
        except ValueError:
            return NotImplemented
        err = ((self.val ** (other.val - 1) * other.val * self.err) ** 2 + (np.log(self.val) * self.val ** other.val * other.err) ** 2) ** 0.5
        val = self.val ** other.val
        return self.__class__(val, err)
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
    def tostring(self, extra_digits=0, plusminus=' +/- ', fmt=None):
        """Make a string representation of the value and its uncertainty.

        Inputs:
        -------
            ``extra_digits``: integer
                how many extra digits should be shown (plus or minus, zero means
                that the number of digits should be defined by the magnitude of
                the uncertainty).
            ``plusminus``: string
                the character sequence to be inserted in place of '+/-'
                including delimiting whitespace.
            ``fmt``: string or None
                how to format the output. Currently only strings ending in 'tex'
                are supported, which render ascii-exponentials (i.e. 3.1415e-2)
                into a format which is more appropriate to TeX.

        Outputs:
        --------
            the string representation.
        """
        if isinstance(fmt, basestring) and fmt.lower().endswith('tex'):
            return re.subn('(\d*)(\.(\d)*)?[eE]([+-]?\d+)', lambda m:(r'$%s%s\cdot 10^{%s}$' % (m.group(1), m.group(2), m.group(4))).replace('None', ''),
                           self.tostring(extra_digits=extra_digits, plusminus=plusminus, fmt=None))[0]
        if isinstance(self.val, numbers.Real):
            try:
                Ndigits = -int(math.floor(math.log10(self.err))) + extra_digits
            except (OverflowError, ValueError):
                return str(self.val) + plusminus + str(self.err)
            else:
                return str(round(self.val, Ndigits)) + plusminus + str(round(self.err, Ndigits))
        return str(self.val) + ' +/- ' + str(self.err)
    def abs(self):
        return self.__class__(np.abs(self.val), self.err)
    def sin(self):
        return self.__class__(np.sin(self.val), np.abs(np.cos(self.val) * self.err))
    def cos(self):
        return self.__class__(np.cos(self.val), np.abs(np.sin(self.val) * self.err))
    def tan(self):
        return self.__class__(np.tan(self.val), np.abs(1 + np.tan(self.val) ** 2) * self.err)
    def sqrt(self):
        return self ** 0.5
    def sinh(self):
        return self.__class__(np.sinh(self.val), np.abs(np.cosh(self.val) * self.err))
    def cosh(self):
        return self.__class__(np.cosh(self.val), np.abs(np.sinh(self.val) * self.err))
    def tanh(self):
        return self.__class__(np.tanh(self.val), np.abs(1 - np.tanh(self.val) ** 2) * self.err)
    def arcsin(self):
        return self.__class__(np.arcsin(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))
    def arccos(self):
        return self.__class__(np.arccos(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))
    def arcsinh(self):
        return self.__class__(np.arcsinh(self.val), np.abs(self.err / np.sqrt(1 + self.val ** 2)))
    def arccosh(self):
        return self.__class__(np.arccosh(self.val), np.abs(self.err / np.sqrt(self.val ** 2 - 1)))
    def arctanh(self):
        return self.__class__(np.arctanh(self.val), np.abs(self.err / (1 - self.val ** 2)))
    def arctan(self):
        return self.__class__(np.arctan(self.val), np.abs(self.err / (1 + self.val ** 2)))
    def log(self):
        return self.__class__(np.log(self.val), np.abs(self.err / self.val))
    def exp(self):
        return self.__class__(np.exp(self.val), np.abs(self.err * np.exp(self.val)))
    def random(self):
        """Sample a random number (array) of the distribution defined by
        mean=`self.val` and variance=`self.err`^2.
        """
        if isinstance(self.val, np.ndarray):
            return np.random.randn(self.val.shape) * self.err + self.val # IGNORE:E1103
        else:
            return np.random.randn() * self.err + self.val
    @classmethod
    def evalfunc(cls, func, *args, **kwargs):
        """Evaluate a function with error propagation.

        Inputs:
        -------
            ``func``: callable
                this is the function to be evaluated. Should return either a
                number or a np.ndarray.
            ``*args``: other positional arguments of func. Arguments which are
                not instances of `ErrorValue` are taken as constants.

            keyword arguments supported:
                ``NMC``: number of Monte-Carlo steps. If not defined, defaults
                to 1000
                ``exceptions_to_retry``: list of exception types to ignore:
                    if one of these is raised the given MC step is repeated once
                    again. Notice that this might induce an infinite loop!
                    The exception types in this list should be subclasses of
                    ``Exception``.
                ``exceptions_to_skip``: list of exception types to skip: if
                    one of these is raised the given MC step is skipped, never
                    to be repeated. The exception types in this list should be
                    subclasses of ``Exception``.


        Output:
        -------
            ``result``: an `ErrorValue` with the result. The error is estimated
                via a Monte-Carlo approach to Gaussian error propagation.
        """
        def do_random(x):
            if isinstance(x, cls):
                return x.random()
            else:
                return x
        if 'NMC' not in kwargs:
            kwargs['NMC'] = 1000
        if 'exceptions_to_skip' not in kwargs:
            kwargs['exceptions_to_skip'] = []
        if 'exceptions_to_repeat' not in kwargs:
            kwargs['exceptions_to_repeat'] = []
        meanvalue = func(*args)
        stdcollector = meanvalue * 0 # this way we get either a number or a np.array
        mciters = 0
        while mciters < kwargs['NMC']:
            try:
                stdcollector += (func(*[do_random(a) for a in args]) - meanvalue) ** 2 #IGNORE:W0142
                mciters += 1
            except Exception as e: #IGNORE:W0703
                if any(isinstance(e, etype) for etype in kwargs['exceptions_to_skip']):
                    kwargs['NMC'] -= 1
                elif any(isinstance(e, etype) for etype in kwargs['exceptions_to_repeat']):
                    pass
                else:
                    raise
        return cls(meanvalue, stdcollector ** 0.5 / (kwargs['NMC'] - 1))
