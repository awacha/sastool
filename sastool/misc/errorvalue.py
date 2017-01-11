# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:18:05 2011

@author: -
"""
import collections
import math
import numbers
import re
from typing import Union, Optional, SupportsFloat, Sequence

import numpy as np

from .arithmetic import ArithmeticBase

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
    val = None
    err = None

    def __init__(self: 'ErrorValue', val: Union[SupportsFloat, np.ndarray, 'ErrorValue', Sequence],
                 err: Optional[Union[SupportsFloat, np.ndarray, 'ErrorValue', Sequence]] = None):
        ArithmeticBase.__init__(self)
        if isinstance(val, numbers.Number):
            self.val = float(val)
            if isinstance(err, numbers.Number):
                self.err = float(err)
            elif err is None:
                self.err = 0.0
            else:
                raise TypeError('err argument is of type %s' % (type(err)))
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
                raise TypeError(
                        "val argument is of type ErrorValue, but its .val attribute is of an unsupported type %s" % type(
                                val.val))
        elif isinstance(val, collections.Sequence):
            if all(isinstance(v, ErrorValue) for v in val):
                self.val = np.array([v.val for v in val])
                self.err = np.array([v.err for v in val])
            elif all(isinstance(v, numbers.Number) for v in val):
                self.val = np.array(val)
                self.err = np.zeros_like(self.val)
            else:
                raise ValueError(
                        'If instantiated with a sequence, all elements of it must either be ErrorValues or numbers.')
        else:
            raise ValueError(
                    'ErrorValue class can hold only Python numbers or numpy ndarrays, got %s!' % type(val))
        assert isinstance(self.val, (float, np.ndarray))
        assert isinstance(self.err, (float, np.ndarray))

    def copy(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(self.val, self.err)

    def __neg__(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(-self.val, self.err)

    def __reciprocal__(self: 'ErrorValue') -> 'ErrorValue':
        """Calculate the reciprocal of this instance"""
        return type(self)(1.0 / self.val, self.err / (self.val * self.val))

    def __getitem__(self: 'ErrorValue', key):
        return type(self)(self.val[key], self.err[key])

    def __iadd__(self: 'ErrorValue', value) -> 'ErrorValue':
        try:
            value = ErrorValue(value)
        except ValueError:
            return NotImplemented
        self.val = self.val + value.val
        self.err = np.sqrt(self.err ** 2 + value.err ** 2)
        return self

    def __imul__(self: 'ErrorValue', value) -> 'ErrorValue':
        try:
            value = ErrorValue(value)
        except ValueError:
            return NotImplemented
        self.err = np.sqrt(self.err * self.err * value.val * value.val +
                           value.err * value.err * self.val * self.val)
        self.val = self.val * value.val
        return self

    def __str__(self) -> str:
        return self.tostring(plusminus=' \xb1 ')

    def __pow__(self: 'ErrorValue', other, modulo=None) -> 'ErrorValue':
        if modulo is not None:
            return NotImplemented
        try:
            other = ErrorValue(other)
        except ValueError:
            return NotImplemented
        err = ((self.val ** (other.val - 1) * other.val * self.err) ** 2 +
               (np.log(np.abs(self.val)) * self.val ** other.val * other.err) ** 2) ** 0.5
        val = self.val ** other.val
        return type(self)(val, err)

    def __repr__(self) -> str:
        return 'ErrorValue(' + repr(self.val) + ' \xb1 ' + repr(self.err) + ')'

    def __float__(self) -> float:
        return float(self.val)

    def __trunc__(self) -> int:
        return int(self.val)

    def __array__(self: 'ErrorValue', dt=None) -> np.ndarray:
        if dt is None:
            return np.array(self.val)
        else:
            return np.array(self.val, dt)

    def tostring(self: 'ErrorValue', extra_digits: int = 0, plusminus: str = ' +/- ', fmt: str = None) -> str:
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
        if isinstance(fmt, str) and fmt.lower().endswith('tex'):
            return re.subn('(\d*)(\.(\d)*)?[eE]([+-]?\d+)',
                           lambda m: (r'$%s%s\cdot 10^{%s}$' % (m.group(1), m.group(2), m.group(4))).replace('None',
                                                                                                             ''),
                           self.tostring(extra_digits=extra_digits, plusminus=plusminus, fmt=None))[0]
        if isinstance(self.val, numbers.Real):
            try:
                Ndigits = -int(math.floor(math.log10(self.err))) + extra_digits
            except (OverflowError, ValueError):
                return str(self.val) + plusminus + str(self.err)
            else:
                return str(round(self.val, Ndigits)) + plusminus + str(round(self.err, Ndigits))
        return str(self.val) + ' +/- ' + str(self.err)

    def abs(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.abs(self.val), self.err)

    def sin(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.sin(self.val), np.abs(np.cos(self.val) * self.err))

    def cos(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.cos(self.val), np.abs(np.sin(self.val) * self.err))

    def tan(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.tan(self.val), np.abs(1 + np.tan(self.val) ** 2) * self.err)

    def sqrt(self: 'ErrorValue') -> 'ErrorValue':
        return self ** 0.5

    def sinh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.sinh(self.val), np.abs(np.cosh(self.val) * self.err))

    def cosh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.cosh(self.val), np.abs(np.sinh(self.val) * self.err))

    def tanh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.tanh(self.val), np.abs(1 - np.tanh(self.val) ** 2) * self.err)

    def arcsin(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arcsin(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))

    def arccos(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arccos(self.val), np.abs(self.err / np.sqrt(1 - self.val ** 2)))

    def arcsinh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arcsinh(self.val), np.abs(self.err / np.sqrt(1 + self.val ** 2)))

    def arccosh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arccosh(self.val), np.abs(self.err / np.sqrt(self.val ** 2 - 1)))

    def arctanh(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arctanh(self.val), np.abs(self.err / (1 - self.val ** 2)))

    def arctan(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.arctan(self.val), np.abs(self.err / (1 + self.val ** 2)))

    def log(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.log(self.val), np.abs(self.err / self.val))

    def exp(self: 'ErrorValue') -> 'ErrorValue':
        return type(self)(np.exp(self.val), np.abs(self.err * np.exp(self.val)))

    def random(self: 'ErrorValue') -> np.ndarray:
        """Sample a random number (array) of the distribution defined by
        mean=`self.val` and variance=`self.err`^2.
        """
        if isinstance(self.val, np.ndarray):
            # IGNORE:E1103
            return np.random.randn(self.val.shape) * self.err + self.val
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
        # this way we get either a number or a np.array
        stdcollector = meanvalue * 0
        mciters = 0
        while mciters < kwargs['NMC']:
            try:
                # IGNORE:W0142
                stdcollector += (func(*[do_random(a)
                                        for a in args]) - meanvalue) ** 2
                mciters += 1
            except Exception as e:  # IGNORE:W0703
                if any(isinstance(e, etype) for etype in kwargs['exceptions_to_skip']):
                    kwargs['NMC'] -= 1
                elif any(isinstance(e, etype) for etype in kwargs['exceptions_to_repeat']):
                    pass
                else:
                    raise
        return cls(meanvalue, stdcollector ** 0.5 / (kwargs['NMC'] - 1))

    def is_zero(self: 'ErrorValue') -> bool:
        return np.abs(self.val) <= np.abs(self.err)

    @classmethod
    def average_independent(cls, lis):
        if not all([isinstance(x, cls) for x in lis]):
            raise ValueError(
                    'All elements of the list should be of the same type: ' + str(cls))
        return cls(sum([x.val / x.err ** 2 for x in lis]) / sum([1 / x.err ** 2 for x in lis]),
                   1 / sum([1 / x.err ** 2 for x in lis]) ** 0.5)

    def __eq__(self, other):
        try:
            other = ErrorValue(other)
        except ValueError:
            return NotImplemented
        return (self.val == other.val) and (self.err == other.err)

    def __lt__(self, other):
        other = ErrorValue(other)
        return (self.val < other.val)

    def __gt__(self, other):
        other = ErrorValue(other)
        return (self.val > other.val)

    def __le__(self, other):
        other = ErrorValue(other)
        return (self.val < other.val)

    def __ge__(self, other):
        other = ErrorValue(other)
        return (self.val > other.val)

    def equivalent(self, other: 'ErrorValue', k=1):
        return abs(self.val - other.val) < k * (self.err ** 2 + other.err ** 2) ** 0.5

    def __format__(self, format_spec=''):
        """Formatting hook.

        Format specification:
        <float_formatspec>[:<pmstring>]

        <pmstring> is a string to put between the value and the error part, including spaces.
        """
        try:
            float_formatspec, pmstring = format_spec.split(':', 1)
        except ValueError:
            if format_spec:
                float_formatspec = format_spec
            else:
                float_formatspec = ''
            pmstring = ' \u00b1 '
        return self.val.__format__(float_formatspec) + pmstring + self.err.__format__(float_formatspec)
