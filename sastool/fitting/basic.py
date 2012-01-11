from fitfunction import FitFunction
import numpy as np

class FFLinear(FitFunction):
    name="Linear"
    argument_info=[('a','slope'),('b','offset')]
    formula="y(x) = a * x + b"
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,a,b):
        return a*x+b

class FFSine(FitFunction):
    name="Sine"
    argument_info=[('a','amplitude'),('omega','circular frequency'),('phi0','phase'),('y0','offset')]
    formula="y(x) = a * sin( omega * x + phi0 ) + y0"
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,a,omega,phi0,y0):
        return a*np.sin(omega*x+phi0)+y0

class FFPolynomial(FitFunction):
    _factory_arguments=[(2,),(3,)]
    name="Polynomial"
    argument_info=[]
    formula=""
    def __init__(self,deg=1):
        FitFunction.__init__(self)
        self.deg=deg
        self.name="Polynomial of order %d"%deg
        self.formula="y(x) = sum_(i=0)^(%d) Ai*x^i" % deg
        self.argument_info=self.argument_info[:]
        for i in range(deg+1):
            self.argument_info.append(('A%d'%i,'Coeff. of the x^%d term'%i))
    def __call__(self,x,*coeffs):
        return np.polynomial.polynomial.polyval(x,coeffs)

class FFPowerlaw(FitFunction):
    _factory_arguments=[(None,),(0,),(1,),(2,)]
    name="Power-law"
    formula="y(x) = A* x^alpha"
    argument_info=[('A','factor'),('alpha','Exponent')]
    def __init__(self,bgorder=None):
        FitFunction.__init__(self)
        self.bgorder=bgorder 
        if bgorder is not None:
            self.name+=" with order #%d background"%bgorder
            self.formula+=" + sum_(i=0)^(%d) Ai*x^i" %bgorder
            self.argument_info=self.argument_info[:]
            for i in range(bgorder+1):
                self.argument_info.append(('A%d'%i,'Coeff. of the x^%d term'%i))
    def __call__(self,x,A,alpha,*bgcoeffs):
        y1=A*np.power(x,alpha)
        if self.bgorder is not None:
            return y1+np.polynomial.polynomial.polyval(x,bgcoeffs)
        return y1

class FFLorentzian(FitFunction):
    name="Lorentzian peak"
    formula="y(x)=A/(1+((x-x0)/sigma)^2)+B"
    argument_info=[('A','Scaling'),('x0','Center'),('sigma','HWHM'),('B','offset')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,x0,sigma,B):
        return A/(1+((x-x0)/sigma)**2)+B

class FFGaussian(FitFunction):
    name="Gaussian peak"
    formula="y(x)=A*exp(-(x-x0)^2)/(2*sigma^2)+B"
    argument_info=[('A','Scaling'),('x0','Center'),('sigma','HWHM'),('B','offset')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,x0,sigma,B):
        return A*np.exp(-(x-x0)**2/(2*sigma**2))+B
