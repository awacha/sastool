from fitfunction import FitFunction
import numpy as np
from scipy.special import sinc, sici

# Helper functions
def Fsphere(q,R):
    return 4*np.pi/q**3*(np.sin(q*R)-q*R*np.cos(q*R))


# Fit functions

class FFGuinier(FitFunction):
    _factory_arguments=[(0,),(1,),(2,)]
    name="Guinier"
    argument_info=[('G','factor'),('Rg','Radius of gyration')]
    formula="y(x) = G * exp(-x^2*Rg^2/3)"
    def __init__(self,qpow=0):
        FitFunction.__init__(self)
        self.qpow=qpow
        if qpow==2:
            self.name="Guinier (thickness)"
            self.formula="y(x) = G * x^(-2) * exp(-x^2*Rg^2)"
        elif qpow==1:
            self.name="Guinier (cross-section)"
            self.formula="y(x) = G * x^(-1) * exp(-x^2*Rg^2/2)"
    def __call__(self,x,G,Rg):
        return G*np.power(x,-self.qpow)*np.exp(-x*x*Rg*Rg/(3-self.qpow))

class FFGuinierPorod(FitFunction):
    name="Guinier-Porod model"
    formula="I(q) = q>sqrt(3*(-alpha)/2)/Rg ? (A*q^alpha) : (G*exp(-q^2*Rg^2/3))"
    argument_info=[('G','Scaling'),('alpha','Power-law exponent'),
                   ('Rg','Radius of gyration')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,G,alpha,Rg):
        q1=np.sqrt(3*(-alpha)/2.)/Rg
        y=np.zeros_like(x)
        A=G*np.exp(alpha/2.)*np.power(q1,-alpha)
        idxGuinier=x<q1
        y[idxGuinier]=G*np.exp(-x[idxGuinier]**2*Rg**2/3.)
        y[-idxGuinier]=A*np.power(x[-idxGuinier],alpha)
        return y

class FF2GuinierPorod(FitFunction):
    name="Guinier-Porod model, sum of two"
    formula="I(q) = GP1(G1,alpha1,Rg1) + GP2(G2,alpha2,Rg2)"
    argument_info=[('G1','Scaling'),('alpha1','Power-law exponent'),
                   ('Rg1','Radius of gyration'),
                   ('G2','Scaling'),('alpha2','Power-law exponent'),
                   ('Rg2','Radius of gyration')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,G1,alpha1,Rg1,G2,alpha2,Rg2):
        q1=np.sqrt(3*(-alpha1)/2.)/Rg1
        q2=np.sqrt(3*(-alpha2)/2.)/Rg2
        y=np.zeros_like(x)
        A1=G1*np.exp(alpha1/2.)*np.power(q1,-alpha1)
        A2=G2*np.exp(alpha2/2.)*np.power(q2,-alpha2)
        idxGuinier1=x<q1
        idxGuinier2=x<q2
        y[idxGuinier1]=G1*np.exp(-x[idxGuinier1]**2*Rg1**2/3.)
        y[-idxGuinier1]=A1*np.power(x[-idxGuinier1],alpha1)
        y[idxGuinier2]+=G2*np.exp(-x[idxGuinier2]**2*Rg2**2/3.)
        y[-idxGuinier2]+=A2*np.power(x[-idxGuinier2],alpha2)
        return y


class FFPorodGuinier(FitFunction):
    name="Porod-Guinier model"
    formula="I(q) = q<sqrt(3*(-alpha)/2)/Rg ? (A*q^alpha) : (G*exp(-q^2*Rg^2/3))"
    argument_info=[('A','Scaling'),('alpha','Power-law exponent'),
                   ('Rg','Radius of gyration')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,alpha,Rg):
        q1=np.sqrt(3*(-alpha)/2.)/Rg
        y=np.zeros_like(x)
        G=A*np.exp(-alpha/2.)*np.power(q1,alpha)
        idxGuinier=x>q1
        y[idxGuinier]=G*np.exp(-x[idxGuinier]**2*Rg**2/3.)
        y[-idxGuinier]=A*np.power(x[-idxGuinier],alpha)
        return y

class FFPorodGuinierPorod(FitFunction):
    name="Porod-Guinier-Porod model"
    formula="I(q) = A*q^(alpha) ; G*exp(-q^2*Rg^2/3) ; B*q^beta smoothly joint"
    argument_info=[('A','Scaling'),('alpha','First power-law exponent'),
                   ('Rg','Radius of gyration'),('beta','Second power-law exponent')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,alpha,Rg,beta):
        q1=np.sqrt(3*(-alpha)/2.)/Rg
        q2=np.sqrt(3*(-beta)/2.)/Rg
        y=np.zeros_like(x)
        G=A*np.exp(-alpha/2.)*np.power(q1,alpha)
        B=G*np.exp(beta/2.)*np.power(q2,-beta)
        idxFirstP=x<q1
        idxSecondP=x>q2
        idxGuinier=(-idxFirstP)&(-idxSecondP)
        y[idxGuinier]=G*np.exp(-x[idxGuinier]**2*Rg**2/3.)
        y[idxFirstP]=A*np.power(x[idxFirstP],alpha)
        y[idxSecondP]=B*np.power(x[idxSecondP],beta)
        return y

class FFPowerlawPlusGuinierPowerlawPlusCorrelationPeak(FitFunction):
    name="Power-law + Guinier-Powerlaw + Correlation Peak"
    formula="I(q)=A*q^(alpha) + (G*exp(-q^2*Rg^2/3) joined smoothly with B*q^beta1)\n+C * exp(-(q-q0)^2/(2*sigma^2))"
    argument_info=[('A','Scaling factor for the first power-law'),
                   ('alpha','Exponent of the first power-law'),
                   ('G','Scaling factor for the Guinier region'),
                   ('Rg','Radius of gyration'),
                   ('beta','Second power-law exponent'),
                   ('C','Scaling for the correlation peak'),
                   ('q0','Correlation peak position'),
                   ('sigma','HWHM of the correlation peak')]
    def __call__(self,x,A,alpha,G,Rg,beta,C,q0,sigma):
        q1=np.sqrt(3*(-beta)/2.)/Rg
        y=np.zeros_like(x)
        B=G*np.exp(beta/2.)*np.power(q1,-beta)
        idxGuinier=x<q1
        y[idxGuinier]=G*np.exp(-x[idxGuinier]**2*Rg**2/3.)
        y[-idxGuinier]=B*np.power(x[-idxGuinier],beta)
        y+=A*np.power(x,alpha)
        y+=C*np.exp(-(x-q0)**2/(2*sigma**2));
        return y
        
class FFTwoPorodGuinierPorod(FitFunction):
    name="Sum of two Porod-Guinier-Porod model"
    formula="I(q) = A1*q^(alpha1) ; G1*exp(-q^2*Rg1^2/3) ; B1*q^beta1 ; + (the same with 2) smoothly joint"
    argument_info=[('A1','Scaling'),('alpha1','First power-law exponent'),
                   ('Rg1','Radius of gyration'),('beta1','Second power-law exponent'),
                   ('A2','Scaling'),('alpha2','First power-law exponent'),
                   ('Rg2','Radius of gyration'),('beta2','Second power-law exponent')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A1,alpha1,Rg1,beta1,A2,alpha2,Rg2,beta2):
        q1=np.sqrt(3*(-alpha1)/2.)/Rg1
        q2=np.sqrt(3*(-beta1)/2.)/Rg1
        y=np.zeros_like(x)
        G=A1*np.exp(-alpha1/2.)*np.power(q1,alpha1)
        B=G*np.exp(beta1/2.)*np.power(q2,-beta1)
        idxFirstP=x<q1
        idxSecondP=x>q2
        idxGuinier=(-idxFirstP)&(-idxSecondP)
        y[idxGuinier]+=G*np.exp(-x[idxGuinier]**2*Rg1**2/3.)
        y[idxFirstP]+=A1*np.power(x[idxFirstP],alpha1)
        y[idxSecondP]+=B*np.power(x[idxSecondP],beta1)

        q1=np.sqrt(3*(-alpha2)/2.)/Rg2
        q2=np.sqrt(3*(-beta2)/2.)/Rg2
        G=A2*np.exp(-alpha2/2.)*np.power(q1,alpha2)
        B=G*np.exp(beta2/2.)*np.power(q2,-beta2)
        idxFirstP=x<q1
        idxSecondP=x>q2
        idxGuinier=(-idxFirstP)&(-idxSecondP)
        y[idxGuinier]+=G*np.exp(-x[idxGuinier]**2*Rg2**2/3.)
        y[idxFirstP]+=A2*np.power(x[idxFirstP],alpha2)
        y[idxSecondP]+=B*np.power(x[idxSecondP],beta2)

        return y

class FFCorrelatedRod(FitFunction):
    name="Rod with correlation"
    argument_info=[('A','scaling'),('L','rod length'),('Rc','Rc parameter'),('sigma','sigma parameter'),('B','coupling strength')]
    formula="y(q) = A* P(q,L)/(1+B*sin(q*Rc)/(q*Rc)*exp(-q^2*sigma^2));\nP(q) = L^2*(2/(q*L)*Si(q*L)-sin(q*L/2)/(q*L/2))"
    def __call__(self,q,A,L,Rc,sigma,B):
        P=L*L*(2/(q*L)*sici(q*L)[0]-sinc(q*L/2./np.pi))
        return A*P/(1+B*sinc(q*Rc/np.pi)*np.exp(-q*q*sigma*sigma)*P)

class FFCorrelatedPowerlaw(FitFunction):
    name="Power-law with correlation"
    argument_info=[('A','scaling'),('alpha','exponent'),('Rc','Rc parameter'),('sigma','sigma parameter'),('B','Coupling strength')]
    formula="y(q) = A*q^-alpha/(1+B*sin(q*Rc)/(q*Rc)*exp(-q^2*sigma^2)*q^-alpha)"
    def __call__(self,q,A,alpha,Rc,sigma,B):
        P=np.power(q,alpha)
        return A*P/(1+B*sinc(q*Rc/np.pi)*np.exp(-q*q*sigma*sigma)*P)
    
class FFPowerlawDamped(FitFunction):
    _factory_arguments=[(0,),(1,),(2,)]
    name="Damped power-law (non-smooth surfaces)"
    formula="y(x) = A* x^alpha * exp(-q^2*sigma^2)"
    argument_info=[('A','factor'),('alpha','Exponent'),('sigma','Thickness of surface')]
    def __init__(self,bgorder=None):
        FitFunction.__init__(self)
        self.bgorder=bgorder 
        if bgorder is not None:
            self.name+=" with order #%d background"%bgorder
            self.formula+=" + sum_(i=0)^(%d) Ai*x^i" %bgorder
            self.argument_info=self.argument_info[:]
            for i in range(bgorder+1):
                self.argument_info.append(('A%d'%i,'Coeff. of the x^%d term'%i))
    def __call__(self,x,A,alpha,sigma,*bgcoeffs):
        y1=A*np.power(x,alpha)*np.exp(-x*x*sigma*sigma)
        if self.bgorder is not None:
            return y1+np.polynomial.polynomial.polyval(x,bgcoeffs)
        return y1

class FFPowerlawGauss(FitFunction):
    name="Power-law plus Gaussian"
    formula="y(x) = A* x^alpha + B*exp(-(x-x0)^2/(2*sigma^2))+C"
    argument_info=[('A','factor of the power-law'),('alpha','power-law exponent'),
                   ('B','factor of the Gaussian'),('x0','center of the Gaussian'),
                   ('sigma','HWHM of the Gaussian'),('C','Constant background')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,alpha,B,x0,sigma,C):
        return A*np.power(x,alpha)+B*np.exp(-(x-x0)**2/(2*sigma**2))+C

class FFLogNormSpherePopulation(FitFunction):
    name="Scattering intensity of a log-normal sphere population"
    formula="y(x)=A*integral_0^inf (p(r)*F^2(x,r)) dr"
    argument_info=[('A','Scaling'),('mu','Expectation of ln(R)'),
                   ('sigma','HWHM of ln(R)'),('Nr','number of R points'),
                  ]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,mu,sigma,Nr):
        # ln R is of a normal distribution with (mu,sigma). We choose lnR_min and
        # lnR_max to be mu-3*sigma and mu+3*sigma.
        Rmin=np.exp(mu-3*sigma)
        Rmax=np.exp(mu+3*sigma)
        R=np.linspace(Rmin,Rmax,Nr)
        weight=1/np.sqrt(2*np.pi*sigma**2*R**2)*np.exp(-(np.log(R)-mu)**2/(2*sigma**2))
        y=np.zeros_like(x)
        for i in range(len(R)):
            y+=weight[i]*Fsphere(x,R[i])
        return y/weight.sum()

