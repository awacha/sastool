'''
Created on Jul 25, 2012

@author: andris
'''

__all__ = []

import numpy as np
import scipy.special
import scipy.stats

from ... import misc


def Fsphere_outer(q, R):
    qR = np.outer(q, R)
    q1 = np.outer(q, np.ones_like(R))
    return 4 * np.pi / q1 ** 3 * (np.sin(qR) - qR * np.cos(qR))


class SizeDistributionFit(object):
    _parameters=[]
    def __init__(self, data, sizemin=None, sizemax=None, Nsize=100, radius=True):
        self._data = data
        self._isradius = radius
        self._fittedcurve=None
        self._sizedistribution=None
        self._fit_statistics=None
        if sizemin is None:
            sizemin = np.pi / self._data.q.max()
        if sizemax is None:
            sizemax = np.pi / self._data.q[self._data.q > 0].min()
        if not self._isradius:
            sizemin = sizemin / 2.
            sizemax = sizemax / 2.
        self._radius = np.linspace(sizemin, sizemax, Nsize)

    def _pdf(self, x, *args):
        raise NotImplementedError('Method _pdf() must be overridden in SizeDistributionFit subclasses!')

    def _fitfunction(self, q, intensityscale, *pdfargs):
        P = self._pdf(self._radius, *pdfargs)
        I = (Fsphere_outer(q, self._radius) ** 2 * np.outer(np.ones_like(q), P))
        return intensityscale * I.sum(1) / P.sum()

    def fit(self, intensityscale, *pdfargs):
        fitresults = misc.nonlinear_leastsquares(self._data.q, self._data.Intensity, self._data.Error,
                                                 self._fitfunction, [intensityscale] + pdfargs)
        self._fit_statistics=fitresults[-1]
        self._fitvalues=fitresults[:-1]
        self._fittedcurve=self._fitfunction(q,*self._fitvalues)
        self._sizedistribution=self._pdf(self._radius,*self._fitvalues)

    def _get_mean(self):
        raise NotImplementedError('Method _get_mean() must be overridden in SizeDistributionFit subclasses!')

    def _get_sigma(self):
        raise NotImplementedError('Method _get_sigma() must be overridden in SizeDistributionFit subclasses!')

class SizeDistributionFit_Gauss(SizeDistributionFit):
    _parameters=[('A','Intensity scaling factor'),
                 (('R0','D0'), ('Mean radius', 'Mean diameter')),
                 (('sigma','sigma'),('HWHM radius', 'HWHM diameter'))]
    def _pdf(self, x, x0, sigma):
        return 1. / (2. * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x - x0) ** 2 / (2. * sigma ** 2))

    def _get_mean(self):
        return self._fitvalues[1]

    def _get_sigma(self):
        return self._fitvalues[2]

    def get_param(self, name):
        pass

class SizeDistributionFit_SchulzZimm(SizeDistributionFit):
    def _pdf(self, x, Ra, k):
        return 1.0/(Ra*scipy.special.gamma(k))*k**k*np.exp(-k*x/Ra)*(x/Ra)**(k-1)

    def _get_mean():
        pass

scipy.stats.distributions.f

