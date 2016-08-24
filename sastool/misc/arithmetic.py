# -*- coding: utf-8 -*-

import abc
from typing import Any

__all__ = ['ArithmeticBase']


class ArithmeticBase(object, metaclass=abc.ABCMeta):
    """A mixin class for defining simple arithmetics with minimal user effort.

    Usage: subclass this object and define ALL of the following methods:
        __imul__(self, value): in-place multiplication of 'self' by 'value'
        __iadd__(self, value): in-place addition of 'value' to 'self'
        __neg__(self): negation, i.e. '-self': should return an instance of the
            same class
        __reciprocal__(self): reciprocal, i.e. '1.0/self'. Should return an instance of
            the same class
        copy(self): should return a deep copy of the current object.

        Note, that __imul__ and __iadd__ too should return the modified version
            of 'self'!

    Methods __add__, __radd__, __sub__, __isub__, __rsub__, __mul__, __rmul__,
        __div__, __rdiv__ and __idiv__ are constructed automatically from the
        given functions (assuming commutative addition and multiplication)
    """

    def __add__(self, value: Any) -> 'ArithmeticBase':
        try:
            obj = self.copy()
        except AttributeError:
            obj = type(self)(self)
        obj.__iadd__(value)
        return obj

    def __radd__(self, value: Any) -> 'ArithmeticBase':
        retval = self + value
        if retval is NotImplemented:
            raise NotImplementedError(
                'addition is not implemented between %s and %s types' % (type(self), type(value)))
        return retval

    def __isub__(self, value: Any) -> 'ArithmeticBase':
        return self.__iadd__(-value)

    def __sub__(self, value: Any) -> 'ArithmeticBase':
        try:
            obj = self.copy()
        except AttributeError:
            obj = type(self)(self)
        obj.__isub__(value)
        return obj

    def __rsub__(self, value: Any) -> 'ArithmeticBase':
        retval = (-self) + value
        if retval is NotImplemented:
            raise NotImplementedError(
                'subtraction is not implemented between %s and %s types' % (type(self), type(value)))
        return retval

    def __mul__(self, value: Any) -> 'ArithmeticBase':
        try:
            obj = self.copy()
        except AttributeError:
            obj = type(self)(self)
        obj = obj.__imul__(value)
        return obj

    def __rmul__(self, value: Any) -> 'ArithmeticBase':
        retval = self * value
        if retval is NotImplemented:
            raise NotImplementedError(
                'multiplication is not implemented between %s and %s types' % (type(self), type(value)))
        return retval

    def __itruediv__(self, value: Any) -> 'ArithmeticBase':
        try:
            value_recip = value.__reciprocal__()
        except AttributeError:
            value_recip = 1.0 / value
        return self.__imul__(value_recip)

    def __truediv__(self, value: Any) -> 'ArithmeticBase':
        try:
            obj = self.copy()
        except AttributeError:
            obj = type(self)(self)
        obj.__itruediv__(value)
        return obj

    def __rtruediv__(self, value: Any) -> 'ArithmeticBase':
        retval = self.__reciprocal__() * value
        if retval is NotImplemented:
            raise NotImplementedError(
                'division is not implemented between %s and %s types' % (type(self), type(value)))
        return retval

    __idiv__ = __itruediv__  # Python2 compatibility
    __div__ = __truediv__  # Python2 compatibility
    __rdiv__ = __rtruediv__  # Python2 compatibility

    @abc.abstractmethod
    def __iadd__(self, value):
        pass

    @abc.abstractmethod
    def __imul__(self, value):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __reciprocal__(self):
        pass

    @abc.abstractmethod
    def copy(self):
        pass
