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
    def copy(self):
        raise NotImplementedError
    def __add__(self, value):
        try:
            obj = self.copy()
        except NotImplementedError:
            obj = type(self)(self)
        obj = obj.__iadd__(value)
        return obj
    def __radd__(self, value):
        retval = self +value
        if retval == NotImplemented:
            raise NotImplementedError('addition is not implemented between %s and %s types' % (type(self), type(value)))
        return retval
    def __isub__(self, value):
        return self.__iadd__(-value)
    def __sub__(self, value):
        try:
            obj = self.copy()
        except NotImplementedError:
            obj = type(self)(self)
        obj = obj.__isub__(value)
        return obj
    def __rsub__(self, value):
        retval = (-self) + value
        if retval == NotImplemented:
            raise NotImplementedError('subtraction is not implemented between %s and %s types' % (type(self), type(value)))
        return retval
    def __mul__(self, value):
        #print "arithmetic.__mul__ starting: ",type(self),type(value)
        try:
            obj = self.copy()
        except NotImplementedError:
            obj = type(self)(self)
        #print "calling imul"
        obj = obj.__imul__(value)
        #print "imul returned"
        return obj
    def __rmul__(self, value):
        #print "arithmetic.__rmul__ starting: ",type(self),type(value)
        retval = self * value
        if retval == NotImplemented:
            #print "arithmetic.__rmul__ not implemented"
            raise NotImplementedError('multiplication is not implemented between %s and %s types' % (type(self), type(value)))
        return retval
    def __idiv__(self, value):
        return self.__imul__(1.0 / value)
    def __div__(self, value):
        try:
            obj = self.copy()
        except NotImplementedError:
            obj = type(self)(self)
        obj = obj.__idiv__(value)
        return obj
    def __rdiv__(self, value):
        retval = self._recip() * value
        if retval == NotImplemented:
            raise NotImplementedError('division is not implemented between %s and %s types' % (type(self), type(value)))
        return retval
    def __iadd__(self, value):
        raise NotImplementedError
    def __imul__(self, value):
        raise NotImplementedError
    def __neg__(self):
        raise NotImplementedError
    def _recip(self):
        raise NotImplementedError
