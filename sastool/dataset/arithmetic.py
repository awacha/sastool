# -*- coding: utf-8 -*-

class ArithmeticBase(object):
    """A mixin class for defining simple arithmetics with minimal user effort.
    
    Usage: subclass this object and define ALL of the following methods:
        __imul__(self, value): in-place multiplication of 'self' by 'value'
        __iadd__(self, value): in-place addition of 'value' to 'self'
        __neg__(self): negation, i.e. '-self': should return an instance of the
            same class
        _recip(self): reciprocal, i.e. '1.0/self'. Should return an instance of
            the same class
        copy(self): should return a deep copy of the current object.
        
        Note, that __imul__ and __iadd__ too should return the modified version
            of 'self'!
            
    Methods __add__, __radd__, __sub__, __isub__, __rsub__, __mul__, __rmul__, 
        __div__, __rdiv__ and __idiv__ are constructed automatically from the
        given functions (assuming commutative addition and multiplication)
    """
    def __add__(self, value):
        obj = self.copy()
        obj += value
        return obj
    def __radd__(self, value):
        return self + value
    def __isub__(self, value):
        return self.__iadd__(-value)
    def __sub__(self, value):
        obj = self.copy()
        obj -= value
        return obj
    def __rsub__(self, value):
        return (-self)+value
    def __mul__(self, value):
        obj = self.copy()
        obj *= value
        return obj
    def __rmul__(self, value):
        return self*value
    def __idiv__(self, value):
        return self.__imul__(1.0/value)
    def __div__(self, value):
        obj = self.copy()
        obj /= value
        return obj
    def __rdiv__(self, value):
        return self._recip()*value
