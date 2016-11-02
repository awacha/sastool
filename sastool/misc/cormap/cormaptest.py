import numpy as np

from .schilling import longest_edge, cormap_pval


def cormap(*intensities):
    """Calculate the correlation map of several 1D arrays.

    Idea from Franke et. al.: Correlation Map, a goodness-of-fit test for
    one-dimensional X-ray scattering spectra. Nature Methods 2015, 12(5),
    pp.419-422 (DOI: 10.1038/nmeth.3358)
    """
    I = np.vstack(intensities)
    Imean = I.mean(axis=0)
    Ired = I - Imean[np.newaxis, :]
    sigma = ((Ired ** 2).sum(axis=0) / (Ired.shape[0] - 1)) ** 0.5
    correl = np.dot(Ired.T, Ired) / (Ired.shape[0] - 1)
    # do something about sigma==0 cases
    sigma2 = np.outer(sigma, sigma)
    assert np.abs((correl == 0) - (sigma2 == 0)).sum() == 0
    sigma2[sigma2 == 0] = 1
    return correl / sigma2


def cormaptest(*intensities):
    """Carry out the cormap test on several one-dimensional
    numpy ndarrays supplied as positional arguments.

    Outputs:
        p-value: probability that the ndarrays are equivalent
        le: longest edge length
        cm: the correlation map matrix.

    Idea from Franke et. al.: Correlation Map, a goodness-of-fit test for
    one-dimensional X-ray scattering spectra. Nature Methods 2015, 12(5),
    pp.419-422 (DOI: 10.1038/nmeth.3358)
    """
    cm = cormap(*intensities)
    le = longest_edge(cm)
    return cormap_pval(cm.shape[0], le), le, cm
