'''
Created on Jul 25, 2012

@author: andris
'''


__all__ = []

import numpy as np


def Fsphere_outer(q, R):
    qR = np.outer(q, R)
    q1 = np.outer(q, np.ones_like(R))
    return 4 * np.pi / q1 ** 3 * (np.sin(qR) - qR * np.cos(qR))


class SizeDistributionFit(object):
    def __init__(self, data, sizemin=None, sizemax=None, Nsize=100, radius=True):
        self._data = data
        self._isradius = radius
        if sizemin is None:
            sizemin=np.pi/self._data.q.max()
        if sizemax is None:
            sizemax=np.pi/self._data.q[self._data.q>0].min()
        if not self._isradius:
            sizemin=sizemin/2.
            sizemax=sizemax/2.
        self._radius=np.linspace(sizemin,sizemax,Nsize)

    def _pdf(self, x, *args):
        raise NotImplementedError('Method _pdf() must be overridden in SizeDistributionFit subclasses!')

    def _fitfunction(self, q, intensityscale, *pdfargs):
        P=self._pdf(self._radius, *pdfargs)
        I = (Fsphere_outer(q, self._radius) ** 2 * np.outer(np.ones_like(q), P))
        return intensityscale * I.sum(1) / P.sum()

    def fit(self, intensityscale, *pdfargs):
        
        pass
class SizeDistributionFit_Gauss(SizeDistributionFit):
    def _pdf(self, x, x0, sigma):
        return 1 / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def _get_mean(self):
        return self._fitvalues[0]

    def _get_sigma(self):
        return self._fitvalues[1]
    
    def 
    