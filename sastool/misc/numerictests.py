__all__ = ['GreaterThan', 'LessThan', 'GreaterOrEqualThan', 'LessOrEqualThan', 'HalfLine', 'Modulo', 'Not', 'InOpenNeighbourhood', 'InClosedNeighbourhood']
 
class HalfLine(object):
    def __init__(self, value, positive=True, closed=True):
        self.value = value
        self.positive = positive
        self.closed = closed
    def __call__(self, cmpvalue):
        if self.closed and self.positive:
            return cmpvalue >= self.value
        elif self.closed and not self.positive:
            return cmpvalue <= self.value
        elif not self.closed and self.positive:
            return cmpvalue > self.value
        elif not self.closed and not self.positive:
            return cmpvalue < self.value

class GreaterThan(HalfLine):
    def __init__(self, value):
        HalfLine.__init__(self, value, True, False)

class LessThan(HalfLine):
    def __init__(self, value):
        HalfLine.__init__(self, value, False, False)

class GreaterOrEqualThan(HalfLine):
    def __init__(self, value):
        HalfLine.__init__(self, value, True, True)

class LessOrEqualThan(HalfLine):
    def __init__(self, value):
        HalfLine.__init__(self, value, False, True)

class Modulo(object):
    def __init__(self, modulus):
        self.modulus = modulus
    def __call__(self, value):
        return value % self.modulus

class InClosedNeighbourhood(object):
    def __init__(self, point, radius):
        self.point = point
        self.radius = radius
    def __call__(self, value):
        return abs(value - self.point) <= self.radius

class InOpenNeighbourhood(object):
    def __init__(self, point, radius):
        self.point = point
        self.radius = radius
    def __call__(self, value):
        return abs(value - self.point) < self.radius

class Not(object):
    def __init__(self, test):
        self.test = test
    def __call__(self, value):
        return not self.test(value)
