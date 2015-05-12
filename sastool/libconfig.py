
# define either 'nm' or 'A' (for Angstroem)
#LENGTH_UNIT='nm'
LENGTH_UNIT='A'

# Planck constant times speed of light
import scipy.constants

def HC():
    hcbase=scipy.constants.codata.value('Planck constant in eV s') *\
           scipy.constants.codata.value('speed of light in vacuum')
    if LENGTH_UNIT=='nm':
        return hcbase*1e9
    elif LENGTH_UNIT=='A':
        return hcbase*1e10
    else:
        raise NotImplementedError('Invalid length unit: '+str(LENGTH_UNIT))

def qunit():
    if LENGTH_UNIT=='nm':
        return 'nm$^{-1}$'
    elif LENGTH_UNIT=='A':
        return '\xc5$^{-1}$'
    else:
        raise NotImplementedError('Invalid length unit: '+str(LENGTH_UNIT))
    
def dunit():
    if LENGTH_UNIT=='nm':
        return 'nm'
    elif LENGTH_UNIT=='A':
        return '\xc5'
    else:
        raise NotImplementedError('Invalid length unit: '+str(LENGTH_UNIT))
        