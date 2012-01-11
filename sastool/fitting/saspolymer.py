from fitfunction import FitFunction
import numpy as np
from scipy.special import gamma, gammainc, sinc


class FFGaussianChain(FitFunction):
    name="Scattering intensity of a Gaussian Chain (Debye function)"
    formula="y(x)=A*2*(exp(-(x*R)^2)-1+(x*R)^2)/(x*R)^4"
    argument_info=[('R','Radius of gyration'),('A','Scaling')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,R,A):
        a=(x*R)**2
        return 2*A*(np.exp(-a)-1+a)/a**2


class FFFloryChain(FitFunction):
    name="Excluded volume chain"
    argument_info=[('A','Scaling'),('Rg','Radius of gyration'),
                   ('nu','Excluded volume parameter')]
    formula="SASFit manual 6. nov. 2010. Equation (3.60b)"
    def __call__(self,x,A,Rg,nu):
        u=x*x*Rg*Rg*(2.*nu+1)*(2.*nu+2)/6.
        return A*(np.power(u,0.5/nu)*gamma(0.5/nu)*gammainc(0.5/nu,u)-
              gamma(1./nu)*gammainc(1./nu,u))/(nu*np.power(u,1./nu));
#        return A*(np.power(u,0.5/nu)*gamma(0.5/nu)-gamma(1/nu)-
#              np.power(u,0.5/nu)*gammaincc(0.5/nu,u)*gamma(0.5/nu)+ 
#            gamma(1/nu)*gammaincc(1/nu,u))/(nu*np.power(u,1/nu));

class FFCorrelatedChain(FitFunction):
    name="Excluded volume chain with correlation effects"
    argument_info=[('A','Scaling'),('Rg','Radius of gyration'),
                   ('nu','Excluded volume parameter'),
                   ('Rc','Parameter for correlations'),
                   ('sigma','Cut-off length for correlations'),
                   ('B','Strength of coupling')]
    formula="y(x) = A * P(q) / (1 + B * sin(q * Rc) / (q * Rc) * e^(-q^2 * sigma^2);\n\
P(q) is the scattering function of a single excluded volume Gaussian chain."
    def __call__(self,x,A,Rg,nu,Rc,sigma,B):
        u=x*x*Rg*Rg*(2*nu+1)*(2*nu+2)/6.
        p=(np.power(u,0.5/nu)*gamma(0.5/nu)*gammainc(0.5/nu,u)-
              gamma(1./nu)*gammainc(1./nu,u))/(nu*np.power(u,1./nu));
        return A*p/(1+p*B*sinc(x*Rc/np.pi)*np.exp(-x*x*sigma*sigma));

class FFHorkay(FitFunction):
    name="Horkay et al. J. Chem. Phys. 125 234904 (9)"
    argument_info=[('F','factor'),
                   ('L','rod length'),
                   ('rc','rod radius'),
                   ('A','factor for the power-law'),
                   ('s','absolute value of the power-law exponent')]
    formula="F/(1+q*L)/(1+q^2*rc^2)+A*q^(-s)"
    def __call__(self,x,F,L,rc,A,s):
        return F/(1+x*L)/(1+x*x*rc*rc)+A*np.power(x,-s)
