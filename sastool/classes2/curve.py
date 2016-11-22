import sys
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..misc.arithmetic import ArithmeticBase
from ..misc.basicfit import findpeak_single
from ..misc.easylsq import nonlinear_odr, FixedParameter
from ..misc.errorvalue import ErrorValue
from ..misc.fitter import Fitter


def errtrapz(x, yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable

    Outputs:
        the error of the integral
    """
    x = np.array(x)
    assert isinstance(x, np.ndarray)
    yerr = np.array(yerr)
    return 0.5 * np.sqrt((x[1] - x[0]) ** 2 * yerr[0] ** 2 +
                         np.sum((x[2:] - x[:-2]) ** 2 * yerr[1:-1] ** 2) +
                         (x[-1] - x[-2]) ** 2 * yerr[-1] ** 2)

class Curve(ArithmeticBase):
    _q_rel_tolerance = 0.05  # relative tolerance when comparing two q values: if 2*(q1-q2)/(q1+q2) > _q_rel_tolerance then q1 ~= q2

    def __init__(self, q: np.ndarray, Intensity: np.ndarray, Error: Optional[np.ndarray] = None,
                 qError: Optional[np.ndarray] = None):
        self.q = q
        assert (Intensity.shape == q.shape)
        self.Intensity = Intensity
        if Error is None:
            Error = np.zeros_like(self.q)
        assert (Error.shape == q.shape)
        self.Error = Error
        if qError is None:
            qError = np.zeros_like(self.q)
        assert qError.shape == q.shape
        self.qError = qError

    def trim(self, qmin=None, qmax=None, Imin=None, Imax=None, isfinite=True):
        idx = np.ones(self.q.shape, np.bool)
        if qmin is not None:
            idx &= (self.q >= qmin)
        if qmax is not None:
            idx &= (self.q <= qmax)
        if Imin is not None:
            idx &= (self.Intensity >= Imin)
        if Imax is not None:
            idx &= (self.Intensity <= Imax)
        if isfinite:
            idx &= np.isfinite(self.q) & np.isfinite(self.Intensity)
        return type(self)(self.q[idx], self.Intensity[idx],
                          self.Error[idx], self.qError[idx])

    def fit(self, fitfunction, parinit, *args, **kwargs):
        """Perform a nonlinear least-squares fit, using sastool.misc.fitter.Fitter()

        Other arguments and keyword arguments will be passed through to the
        __init__ method of Fitter. For example, these are:
        - lbounds
        - ubounds
        - ytransform
        - loss
        - method

        Returns: the final parameters as ErrorValue instances, the stats
            dictionary and the fitted curve instance of the same type as
            this)
        """
        fitter = Fitter(fitfunction, parinit, self.q, self.Intensity, self.qError, self.Error, *args, **kwargs)
        fittable = [not isinstance(p, FixedParameter) for p in parinit]
        fixedvalues = [[parinit[i], None][fittable[i]] for i in range(len(fittable))]
        fitter.fixparameters(fixedvalues)
        fitter.fit()
        pars = fitter.parameters()
        uncs = fitter.uncertainties()
        stats = fitter.stats()
        results = [ErrorValue(p, u) for p, u in zip(pars, uncs)] + [stats, type(self)(self.q, stats['func_value'])]
        return results

    def odr(self, fitfunction, parinit, *args, **kwargs):
        result = list(
            nonlinear_odr(self.q, self.Intensity, self.qError, self.Error, fitfunction, parinit, *args, **kwargs))
        result.append(type(self)(self.q, result[-1]['func_value'], np.zeros_like(self.q), np.zeros_like(self.q)))
        return result

    def peakfit(self, peaktype='Gaussian', signs=(1, -1)):
        result = list(
            findpeak_single(self.q, self.Intensity, self.Error, curve=peaktype, return_stat=True, signs=signs))
        result.append(type(self)(self.q, result[-1]['func_value']))
        return result

    def sanitize(self):
        idx = (self.q > 0) & np.isfinite(self.Intensity) & np.isfinite(self.q)
        return type(self)(self.q[idx], self.Intensity[idx], self.Error[idx], self.qError[idx])

    def __len__(self):
        return len(self.q)

    def loglog(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt.gca()
        c = self.trim(qmin=sys.float_info.epsilon, Imin=sys.float_info.epsilon, isfinite=True)
        return ax.loglog(c.q, c.Intensity, *args, **kwargs)

    def plot(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt.gca()
        c = self.trim(isfinite=True)
        return ax.plot(c.q, c.Intensity, *args, **kwargs)

    def semilogx(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt.gca()
        c = self.trim(qmin=sys.float_info.epsilon, isfinite=True)
        return ax.semilogx(c.q, c.Intensity, *args, **kwargs)

    def semilogy(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt.gca()
        c = self.trim(Imin=sys.float_info.epsilon, isfinite=True)
        return ax.semilogy(c.q, c.Intensity, *args, **kwargs)

    def errorbar(self, *args, **kwargs):
        if 'axes' in kwargs:
            ax = kwargs['axes']
            del kwargs['axes']
        else:
            ax = plt.gca()
        return ax.errorbar(self.q, self.Intensity, self.Error, self.qError, *args, **kwargs)

    @staticmethod
    def compare_qranges(curve1, curve2, rel_tolerance=0.05, verbose=True):
        if len(curve1.q) != len(curve2.q):
            if verbose:
                print('Length of curves differ: {:d} and {:d}'.format(len(curve1.q), len(curve2.q)))
            return False
        different_index = []
        nonfinite_index = []
        for i in range(len(curve1.q)):
            if curve1.q[i] + curve2.q[i] == 0:
                continue
            if not np.isfinite(curve1.q[i]) or not np.isfinite(curve2.q[i]):
                nonfinite_index.append(i)
            elif abs(curve1.q[i] - curve2.q[i]) / (curve1.q[i] + curve2.q[i]) * 2 >= rel_tolerance:
                different_index.append(i)
                if not verbose:
                    return False
        if different_index:
            if verbose:
                print('Differring points (index, curve1.q, curve2.q):')
                for d in different_index:
                    print('   {:d}, {:f}, {:f}'.format(d, curve1.q[d], curve2.q[d]))
            return False
        else:
            if verbose:
                print('The two q-ranges are compatible within {:f} relative tolerance'.format(rel_tolerance))
            return True

    def _check_q_compatible(self, other):
        return self.compare_qranges(self, other, self._q_rel_tolerance, False)

    def __iadd__(self, other):
        if isinstance(other, Curve):
            self._check_q_compatible(other)
            self.q = 0.5 * (self.q + other.q)
            self.qError = (self.qError ** 2 + other.qError ** 2) ** 0.5 / 4.
            self.Intensity = self.Intensity + other.Intensity
            self.Error = (self.Error ** 2 + other.Error ** 2) ** 0.5
        elif isinstance(other, ErrorValue):
            self.Intensity = self.Intensity + other.val
            self.Error = (self.Error ** 2 + other.err ** 2) ** 0.5
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, np.ndarray):
            self.Intensity = self.Intensity + other
        else:
            return NotImplemented
        return self

    def __imul__(self, other):
        if isinstance(other, Curve):
            self._check_q_compatible(other)
            self.q = 0.5 * (self.q + other.q)
            self.qError = (self.qError ** 2 + other.qError ** 2) ** 0.5 / 4.
            self.Error = (self.Error ** 2 * other.Intensity ** 2 + other.Error ** 2 * self.Intensity ** 2) ** 0.5
            self.Intensity = self.Intensity * other.Intensity
        elif isinstance(other, ErrorValue):
            self.Intensity = self.Intensity * other.val
            self.Error = (self.Error ** 2 * other.val ** 2 + self.Intensity ** 2 * other.err ** 2) ** 0.5
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, np.ndarray):
            self.Intensity = self.Intensity * other
            self.Error = self.Error * other
        else:
            return NotImplemented
        return self

    def __reciprocal__(self):
        return type(self)(self.q, 1 / self.Intensity, self.Error / self.Intensity ** 2, self.qError)

    def __neg__(self):
        return type(self)(self.q, -self.Intensity, self.Error, self.qError)

    def copy(self):
        return type(self)(self.q, self.Intensity, self.Error, self.qError)

    def save(self, filename):
        data = np.stack((self.q, self.Intensity, self.Error, self.qError), 1)
        np.savetxt(filename, data, header='q\tIntensity\tError\tqError')

    def __getitem__(self, item):
        return type(self)(self.q[item], self.Intensity[item], self.Error[item], self.qError[item])

    def interpolate(self, newq, **kwargs):
        return type(self)(newq,
                          np.interp(newq, self.q, self.Intensity, **kwargs),
                          np.interp(newq, self.q, self.Error, **kwargs),
                          np.interp(newq, self.q, self.qError, **kwargs))

    def momentum(self, exponent=1, errorrequested=True):
        """Calculate momenta (integral of y times x^exponent)
        The integration is done by the trapezoid formula (np.trapz).

        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned (true Gaussian
                error-propagation of the trapezoid formula)
        """
        y = self.Intensity * self.q ** exponent
        m = np.trapz(y, self.q)
        if errorrequested:
            err = self.Error * self.q ** exponent
            dm = errtrapz(self.q, err)
            return ErrorValue(m, dm)
        else:
            return m

    @classmethod
    def merge(cls, first, last, qsep=None):
        if not (isinstance(first, cls) and isinstance(last, cls)):
            raise ValueError('Cannot merge types %s and %s together, only %s is supported.' % (
                type(first), type(last), cls))
        if qsep is not None:
            first = first.trim(qmax=qsep)
            last = last.trim(qmin=qsep)
        data = np.concatenate((first.as_structarray(), last.as_structarray()))
        data = np.sort(data, order='q')
        return cls(data['q'], data['Intensity'], data['Error'], data['qError'])

    def as_structarray(self):
        data = np.zeros(len(self), dtype=[('q', np.double),
                                          ('Intensity', np.double),
                                          ('Error', np.double),
                                          ('qError', np.double)])
        data['q'] = self.q
        data['Intensity'] = self.Intensity
        data['Error'] = self.Error
        data['qError'] = self.qError
        return data

    def scalefactor(self, other, qmin=None, qmax=None, Npoints=None):
        """Calculate a scaling factor, by which this curve is to be multiplied to best fit the other one.

        Inputs:
            other: the other curve (an instance of GeneralCurve or of a subclass of it)
            qmin: lower cut-off (None to determine the common range automatically)
            qmax: upper cut-off (None to determine the common range automatically)
            Npoints: number of points to use in the common x-range (None defaults to the lowest value among
                the two datasets)

        Outputs:
            The scaling factor determined by interpolating both datasets to the same abscissa and calculating
                the ratio of their integrals, calculated by the trapezoid formula. Error propagation is
                taken into account.
        """
        if qmin is None:
            qmin = max(self.q.min(), other.q.min())
        if qmax is None:
            xmax = min(self.q.max(), other.q.max())
        data1 = self.trim(qmin, qmax)
        data2 = other.trim(qmin, qmax)
        if Npoints is None:
            Npoints = min(len(data1), len(data2))
        commonx = np.linspace(
                max(data1.q.min(), data2.q.min()), min(data2.q.max(), data1.q.max()), Npoints)
        data1 = data1.interpolate(commonx)
        data2 = data2.interpolate(commonx)
        return nonlinear_odr(data1.Intensity, data2.Intensity, data1.Error, data2.Error, lambda x, a: a * x, [1])[0]

    def unite(self, other, qmin=None, qmax=None, qsep=None,
              Npoints=None, scaleother=True, verbose=False, return_factor=False):
        if not isinstance(other, type(self)):
            raise ValueError(
                    'Argument `other` should be an instance of class %s' % type(self))
        if scaleother:
            factor = other.scalefactor(self, qmin, qmax, Npoints)
            retval = type(self).merge(self, factor * other, qsep)
        else:
            factor = self.scalefactor(other, qmin, qmax, Npoints)
            retval = type(self).merge(factor * self, other, qsep)
        if verbose:
            print("Uniting two datasets.")
            print("   xmin   : ", qmin)
            print("   xmax   : ", qmax)
            print("   xsep   : ", qsep)
            print("   Npoints: ", Npoints)
            print("   Factor : ", factor)
        if return_factor:
            return retval, factor
        else:
            return retval

    @classmethod
    def average(cls, *curves):
        q = np.stack([c.q for c in curves], axis=1)
        I = np.stack([c.Intensity for c in curves], axis=1)
        dq = np.stack([c.qError for c in curves], axis=1)
        dI = np.stack([c.Error for c in curves], axis=1)
        # the biggest problem here is qError==0 or Error==0
        for a, name in [(dq, 'q'), (dI, 'I')]:
            if (a == 0).sum():
                warnings.warn('Some %s errors are zeros, trying to fix them.' % name)
                for i in range(q.shape[1]):
                    try:
                        a[i, :][a[i, :] == 0] = a[i, :][a[i, :] != 0].mean()
                    except:
                        a[i, :][a[i, :] == 0] = 1
        I = (I / dI ** 2).sum(axis=1) / (1 / dI ** 2).sum(axis=1)
        dI = 1 / (1 / dI ** 2).sum(axis=1) ** 0.5
        q = (q / dq ** 2).sum(axis=1) / (1 / dq ** 2).sum(axis=1)
        dq = 1 / (1 / dq ** 2).sum(axis=1) ** 0.5
        return cls(q, I, dI, dq)

    @classmethod
    def new_from_file(cls, filename, *args, **kwargs):
        data = np.loadtxt(filename, *args, **kwargs)
        q = data[:, 0]
        I = data[:, 1]
        try:
            E = data[:, 2]
        except IndexError:
            E = None
        try:
            qE = data[:, 3]
        except IndexError:
            qE = None
        return cls(q, I, E, qE)
