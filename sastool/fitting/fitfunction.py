# -*- coding: utf-8 -*-
"""
Fit functions and transformations.

To extend this file, please make subclasses of FitFunction or Transform.

How to do this?

FitFunction:
    you have to override 'name', 'formula', 'argument_info', '__init__' and '__call__'

Transform:
    you have to override 'name' and 'do_transform', optionally 'xlabel()' and 'ylabel()'

The function Factory('FitFunction') or Factory('Transform') will create a list
    of instances of the subclasses of the specified functions.

In both cases, you can add a class-level attribute _factory_arguments, which
    should either be None (in this case the subclass is ignored by Factory()),
    or a list of tuples. In the latter case, as many instances will be created
    from the fit function subclass, as many tuples are in this list. The tuples
    will be expanded to supply the argument list for the constructors (__init__)

Conventions: names for subclasses of FitFunction should start with 'FF'. Names
    for subclasses of Transform should start with 'Transform'.

@author: Andras Wacha
"""

import numpy as np
from scipy.special import *
import _fitfunction

class FitFunction(object):
    """Abstract base class for fitting functions.
    
    The following class attributes are defined:
        argument_info: a list of tuples of strings in the form:
            [(arg1_shortname, arg1_descr), (arg2_shortname, arg2_descr)...]
        name: unique name of this function
        formula: free-form string describing the formula defining this function.
        defaultargs: list of the default values for all arguments or an empty
            list
        transparms: 
        _noneparams:
        args:
        _factory_arguments: if __init__ accepts arguments, this can be a list
            of tuples. The function Factory() will create instances from each
            subclass of FitFunction, using each tuple in this list in turn as
            the arguments for the constructor.
    """
    argument_info=[]
    name=""
    formula=""
    defaultargs=[]
    transparams=[]
    _noneparams=None
    args=[]
    def __init__(self,function=None):
        if function is not None:
            self._function=function
    def __call__(self,x,*args,**kwargs):
        """This is a generic __call__ method. This should be overridden via either
        of these two ways:
            1) define the <function> keyword argument of the constructor, or
            2) subclass and overload this method.
            
        In either case, the function should accept at least one argument, the 
        abscissa (i.e. "x") vector. Fittable arguments should be positional
        parameters, all others should be keyword arguments. I.e. the following
        function should qualify:
            def func(x,a,omega,phi,y0):
                return a*np.sin(omega*x+phi)+y0
        
        """
        if self._function is not None:
#            print "__call__(): self._function"
#            print "original args:",args
            if len(self.transparams)>0:
                args1=self.transparams[:]
                if self._noneparams is None:
#                    print "Calculating self._noneparams"
                    self._noneparams=[i for i in range(len(self.transparams)) if self.transparams[i] is None]
#                print "self._noneparams",self._noneparams
                for i,j in zip(self._noneparams,range(len(args))):
#                    print "Substituting argument #%d by %s" % (i,args[j])
                    args1[i]=args[j]
            else:
                args1=args
#            print "argument list: ",args1
            return self._function(x,*args1,**kwargs)
        raise NotImplementedError
    def init_arguments(self,*args):
        if len(self.defaultargs)==len(self.argument_info):
            return self.defaultargs
        else:
            return [0]*len(self.argument_info)
    def savedefaults(self,args):
        self.defaultargs=args
    def numargs(self):
        return len(self.argument_info)
    def specialize(self,**kwargs):
        """Create a specialized instance of this function.
        
        Inputs:
            <paramname>=<paramvalue> pairs
        
        Output:
            specialized function instance
            
        Notes:
            each keyword argument to this method which is a parameter for this
            fit function gets fixed: i.e. it is removed from the argument list
            of the new instance.
        """
#        print "Specializing..."
        transparams=[None]*len(self.argument_info)
        arginfo1=[]
        formula=self.formula
        for i in range(len(self.argument_info)):
            k=self.argument_info[i][0]
            if k in kwargs.keys():
                transparams[i]=kwargs[k]
                formula='%s | %s = %g' % (formula,k,kwargs[k])
            else:
                arginfo1.append(self.argument_info[i])
        #transparams will have None at fittable arguments and a defined value at
        # the fixed arguments
#        print "Transparams:",transparams
        x=self.copy()
        x.transparams=transparams
        x._function=self
        x.argument_info=arginfo1
        x.formula=formula
        return x
    def funcinfo(self):
        return {'funcname':self.name,'paramnames':['%s (%s)'%(x[0],x[0]) for x in self.argument_info],'plotmethod':'plot','formula':self.formula}
    def copy(self):
        x=FitFunction()
        x.argument_info=self.argument_info[:]
        x.name=self.name
        x.formula=self.formula
        x.defaultargs=self.defaultargs[:]
        x.transparams=self.transparams[:]
        return x
    def pushargs(self,args):
        if hasattr(self,'args'):
            if len(self.args)>0:
                if all([x==y for x,y in zip (self.args[-1],args)]):
                    #print "Skipping pushargs (len(args)=%d)"%len(self.args)
                    return
            self.args.append(args)
        else:
            self.args=[args]
    def popargs(self):
        if hasattr(self,'args'):
            return self.args.pop()
        else:
            return self.defaultargs
    def getargs(self,idx):
        if hasattr(self,'args'):
            return self.args[idx]
        else:
            return self.defaultargs
            
class Transform(object):
    name='Transform base class'
    def __init__(self):
        pass
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        raise NotImplementedError('Transform is an abstract class. Please subclass it and override the do_transform method.')
    def __call__(self,*args,**kwargs):
        return self.do_transform(*args,**kwargs)
    def xlabel(self,unit=u'1/\xc5'):
        return u'q (%s)' %unit
    def ylabel(self,unit=u'1/cm'):
        return u'y (%s)' % unit
    def __str__(self):
        return self.name
    __unicode__=__str__
    def __repr__(self):
        return '<Transform (%s)>'%self.name

        
class FFFloryChain(FitFunction):
    name="Excluded volume chain"
    argument_info=[('A','Scaling'),('Rg','Radius of gyration'),
                   ('nu','Excluded volume parameter')]
    formula="SASFit manual 6. nov. 2010. Equation (3.60b)"
    def __call__(self,x,A,Rg,nu):
        u=x*x*Rg*Rg*(2*nu+1)*(2*nu+2)/6.
        l=1.0/nu;
        return A*(np.power(u,0.5/nu)*gamma(0.5/nu)*gammainc(0.5/nu,u)-
              gamma(1./nu)*gammainc(1./nu,u))/(nu*np.power(u,1/nu));
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
        l=1.0/nu;
        p=(np.power(u,0.5/nu)*gamma(0.5/nu)*gammainc(0.5/nu,u)-
              gamma(1./nu)*gammainc(1./nu,u))/(nu*np.power(u,1/nu));
        return A*p/(1+p*B*sinxdivx(x*Rc)*np.exp(-x*x*sigma*sigma));
    
class FFLinear(FitFunction):
    name="Linear"
    argument_info=[('a','slope'),('b','offset')]
    formula="y(x) = a * x + b"
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,a,b):
        return a*x+b

class FFCorrelatedRod(FitFunction):
    name="Rod with correlation"
    argument_info=[('A','scaling'),('L','rod length'),('Rc','Rc parameter'),('sigma','sigma parameter'),('B','coupling strength')]
    formula="y(q) = A* P(q,L)/(1+B*sin(q*Rc)/(q*Rc)*exp(-q^2*sigma^2));\nP(q) = L^2*(2/(q*L)*Si(q*L)-sin(q*L/2)/(q*L/2))"
    def __call__(self,q,A,L,Rc,sigma,B):
        P=L*L*(2/(q*L)*sici(q*L)[0]-sinc(q*L/2.))
        return A*P/(1+B*sinc(q*Rc)*exp(-q*q*sigma*sigma)*P)

class FFCorrelatedPowerlaw(FitFunction):
    name="Power-law with correlation"
    argument_info=[('A','scaling'),('alpha','exponent'),('Rc','Rc parameter'),('sigma','sigma parameter'),('B','Coupling strength')]
    formula="y(q) = A*q^-alpha/(1+B*sin(q*Rc)/(q*Rc)*exp(-q^2*sigma^2)*q^-alpha)"
    def __call__(self,q,A,alpha,Rc,sigma,B):
        P=np.power(q,alpha)
        return A*P/(1+B*sinc(q*Rc)*exp(-q*q*sigma*sigma)*P)
        
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


class FFSine(FitFunction):
    name="Sine"
    argument_info=[('a','amplitude'),('omega','circular frequency'),('phi0','phase'),('y0','offset')]
    formula="y(x) = a * sin( omega * x + phi0 ) + y0"
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,a,omega,phi0,y0):
        return a*np.sin(omega*x+phi0)+y0

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
    
class FFPowerlaw(FitFunction):
    _factory_arguments=[(0,),(1,),(2,)]
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

class FFBilayer3Gauss(FitFunction):
    name="Bilayer form factor from 3 Gaussians"
    formula="y(x)=Bilayer3Gauss(x,R0,dR,NR,thead,ttail,rhohead,rhotail)"
    argument_info=[('R0','Mean radius of vesicles'),('dR','HWHM of R'),('NR','R discretization'),
                   ('thead','Thickness of head groups'),
                   ('ttail','Length of tail region'),('rhohead','SLD of head groups'),
                   ('rhotail','SLD of tail'),
                   ('bg','Constant background')]
    defaultargs=[200,50,100,3,3.5,380,304,0]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,R0,dR,NR,thead,ttail,rhohead,rhotail,bg):
        whead=thead*np.sqrt(2*np.pi)
        wtail=ttail*np.sqrt(2*np.pi)
        r=np.linspace(R0-dR*3,R0+dR*3,NR);
        if len(r)==1 or dR==0:
            weight=np.ones_like(r)
        else:
            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2);
        I=np.zeros_like(x)
        for i in range(len(r)):
            R=r[i]
            Rout=R-whead/2
            Rtail=R-(whead+wtail/2)
            Rin=R-(whead+wtail+whead/2)
            
            Fbin=rhohead*FGaussianShell(x,Rin,thead)
            Fbout=rhohead*FGaussianShell(x,Rout,thead)
            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
            I+=(Fbin+Fbout+Fbtail)**2*weight[i]
        return I/weight.sum()+bg;

class FFBilayer3Step(FitFunction):
    name="Bilayer form factor from 3 Step functions"
    formula="y(x)=Bilayer3Step(x,R0,dR,NR,whead,wtail,rhohead,rhotail)"
    argument_info=[('R0','Mean radius of vesicles'),('dR','HWHM of R'),('NR','R discretization'),
                   ('whead','Thickness of head groups'),
                   ('wtail','Length of tail region'),('rhohead','SLD of head groups'),
                   ('rhotail','SLD of tail'),
                   ('bg','Constant background')]
    defaultargs=[200,50,100,3,3.5,380,304,0]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,R0,dR,NR,whead,wtail,rhohead,rhotail,bg):
        r=np.linspace(R0-dR*3,R0+dR*3,NR);
        if len(r)==1 or dR==0:
            weight=np.ones_like(r)
        else:
            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2);
        I=np.zeros_like(x)
        for i in range(len(r)):
            R=r[i]
            Rout=R-whead/2
            Rtail=R-whead-wtail/2
            Rin=R-whead-wtail-whead/2
            Fbin=Fsphere(x,Rin+whead/2)-Fsphere(x,Rin-whead/2)
            Fbout=Fsphere(x,Rout+whead/2)-Fsphere(x,Rout-whead/2)
            Fbtail=Fsphere(x,Rtail+wtail/2)-Fsphere(x,Rtail-wtail/2)
            I+=(rhohead/whead*(Fbin+Fbout)+rhotail/wtail*Fbtail)**2*weight[i]
        return I/weight.sum()+bg;
        
class FFMicelleLogNorm(FitFunction):
    name="Micelle scattered intensity from step functions, with log-normal size distribution"
    formula="y(x)=Micelle(x,Factor,rhohead,rhotail,rhoPEG,whead,dwhead,wtail,dwtail,wPEG,dwPEG,N,bg)"
    argument_info=[('Factor','Scaling factor'),
                   ('Rout','Mean value of the outer radius'),
                   ('sigma','Sigma parameter of the log-normal distribution of radii'),
                   ('rhohead','SLD of head groups'),
                   ('rhotail','SLD of hydrocarbon chains'),
                   ('rhoPEG','SLD of PEG chains'),
                   ('whead','thickness of head groups'),
                   ('shead','stretching weight of the head groups'),
                   ('wtail','length of the hydrocarbon chain region'),
                   ('stail','stretching weight of the tail'),
                   ('wPEG','length of the PEG chains'),
                   ('sPEG','stretching weight of the PEG chains'),
                   ('N','Number of MC steps'),
                   ('bg','Constant background')]
    defaultargs=[1, 5, 0.001, 0.443-0.333, 0.302-0.333, 0.3385-0.333, 0.5, 
                 1, 1.6, 1, 4.6, 1, 100, 0]
    def __call__(self, x, Factor, Rout, sigma, rhohead, rhotail, rhoPEG, whead, 
                 shead, wtail, stail, wPEG, sPEG, N, bg):
        # N samples from a log-normal distribution
        R=np.random.lognormal(np.log(Rout),sigma,N)
        #normalize the stretching weights
        sums=float(shead*whead+stail*wtail+sPEG*wPEG)
        shead=shead*whead/sums
        stail=stail*wtail/sums
        sPEG=sPEG*wPEG/sums
        #divide the difference between R and (whead+wtail+wPEG) among the three
        # leaflets with respect to the stretching weights
        residualR=R-(whead+wtail+wPEG)
        whead=whead+residualR*shead
        wPEG=wPEG+residualR*sPEG
        #wtail=wtail+residualR*stail # no need for this quantity later
        # PEG layer, as a hollow sphere
        F=rhoPEG*(Fsphere_outer(x,R)-Fsphere_outer(x,R-wPEG))
        # head groups, hollow sphere
        F+=rhohead*(Fsphere_outer(x,R-wPEG)-Fsphere_outer(x,R-wPEG-whead))
        # tail groups, full sphere
        F+=rhotail*(Fsphere_outer(x,R-whead-wPEG))
        # intensity
        return Factor*(F.sum(1)/float(N))**2+bg
        
class FFGaussianChain(FitFunction):
    name="Scattering intensity of a Gaussian Chain (Debye function)"
    formula="y(x)=A*2*(exp(-(x*R)^2)-1+(x*R)^2)/(x*R)^4"
    argument_info=[('R','Radius of gyration'),('A','Scaling')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,R,A):
        a=(x*R)**2
        return 2*A*(np.exp(-a)-1+a)/a**2

class FFGaussian(FitFunction):
    name="Gaussian peak"
    formula="y(x)=A*exp(-(x-x0)^2)/(2*sigma^2)+B"
    argument_info=[('A','Scaling'),('x0','Center'),('sigma','HWHM'),('B','offset')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,x0,sigma,B):
        return A*np.exp(-(x-x0)**2/(2*sigma**2))+B

def FGaussianShell(q,R,sigma):
    """Scattering factor of a spherical shell with a radial Gaussian el.density profile
    
    Inputs:
        q: q values
        R: radius
        sigma: HWHM of the radial Gauss profile
        
    Outputs:
        the form factor (same format as q), with the convention F(q=0)=V
        
    Notes:
        Gradzielski & al (J. Phys. Chem. 99 pp 13232-8)
    """
    return 4*np.pi*np.sqrt(2*np.pi)*sigma/q*np.exp(-(q*sigma)**2/2)*(R*np.sin(q*R)+q*sigma**2*np.cos(q*R))

def VGaussianShell(R,t):
    """Volume of a Gaussian Shell
    """
    return 4*np.pi*np.sqrt(2*np.pi)*t*(R**2+t**2)

def FGaussianChain(q,Rg):
    """Form factor (sqrt!!) of a Gaussian chain
    
    Inputs:
        q: q values
        Rg: radius of gyration
        
    Outputs:
        the form factor, 1-exp(-(q*Rg)**2)/(q*Rg), F(q=0)=1
    """
    x=(q*Rg)**2
    if np.isscalar(x):
        if x==0:
            return 1
        else:
            return (1-np.exp(-x))/x
    y=np.ones_like(x)
    idxok=x!=0
    y[idxok]=(1-np.exp(-x[idxok]))/(x[idxok])
    return y

def F2GaussianChain(q,Rg):
    """Squared form factor (self interference term) of a Gaussian chain
    
    Inputs:
        q: q values
        Rg: radius of gyration
        
    Outputs:
        the squared form factor, 2*(exp(-x)-1+x)/x^2, i.e. the Debye function,
            where x=(q*Rg)^2. The convention F(q=0)=1 is used.
    """
    x=(q*Rg)**2
    if np.isscalar(x):
        if x==0:
            return 1
        else:
            return 2*(np.exp(-x)-1+x)/x**2
    y=np.ones_like(x)
    idxok=x!=0
    y[idxok]=2*(np.exp(-x[idxok])-1+x[idxok])/(x[idxok])**2
    return y

def sinxdivx(x):
    """Evaluate sin(x)/x safely (x->0 limit is 1)
    
    """
    if np.isscalar(x):
        if x==0:
            return 1
        else:
            return np.sin(x)/x
    y=np.ones_like(x)
    idxok=x!=0
    y[idxok]=np.sin(x[idxok])/x[idxok]
    return y

class FGaussianBilayerExact(FitFunction):
    name="PEG-decorated bilayer, Gauss profiles"
    formula=""
    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
                   ('NR','Number of R points'),
                   ('rhohead','SLD of lipid head region'),
                   ('rhotail','SLD of lipid tail region'),
                   ('rhoPEG','SLD of PEG chain region'),
                   ('zhead','distance of the head groups from the center of the bilayer'),
                   ('zPEG','distance of the PEG chains from the center of the bilayer'),
                   ('thead','HWHM of the head group profile'),
                   ('ttail','HWHM of the head group profile'),
                   ('tPEG','HWHM of the head group profile'),
                   ('bg','Constant background')
                  ]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,R0,dR,NR,rhohead,rhotail,rhoPEG,zhead,zPEG,thead,ttail,tPEG,bg):
        r=np.linspace(0,R0+3*dR,NR)
        if len(r)==1 or dR==0:
            weight=np.ones_like(r)
        else:
            weight=np.exp(-(r-R0)**2/(2*dR**2))
        I=np.zeros_like(x)
        for i in range(len(r)):
            #dimenziok szamolasa
            R=r[i] # aktualis liposzomameret (legkulso sugar, PEG reteget leszamitva)
            # D=whead*2+wtail # A kettosreteg vastagsaga
            # A liposzomat alkoto harom Gauss-reteg sulyponti sugarai
            Rin=R-zhead 
            Rout=R+zhead
            Rtail=R
            RPEGout=R+zPEG
            RPEGin=R-zPEG
            Fbin=rhohead*FGaussianShell(x,Rin,thead)
            Fbout=rhohead*FGaussianShell(x,Rout,thead)
            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
            FbPEGin=rhoPEG*FGaussianShell(x,RPEGin,tPEG)
            FbPEGout=rhoPEG*FGaussianShell(x,RPEGout,tPEG)
            thisI=(Fbin+Fbout+Fbtail+FbPEGin+FbPEGout)**2*weight[i]
            I+=thisI
        return I/weight.sum()*A+bg
        
class FGaussianBilayerPEG(FitFunction):
    name="Gaussian Bilayer with PEG"
    formula="Brzustowicz & Brunger, J. Appl. Cryst. 38 pp 126-31"
    argument_info=[('A','Scaling'),
                   ('R0','Mean radius'),
                   ('rhotail','SLD of CH region'),
                   ('rhoPEG','SLD of PEG chains'),
                   ('sigmahead','SLD of head region'),
                   ('sigmatail','SLD of tail region'),
                   ('sigmaPEG','HWHM of PEG chains'),
                   ('zhead','position of head region'),
                   ('zPEG','position of the PEG chains'),
                   ('bg','Constant background')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,R0,rhotail,rhoPEG,sigmahead,sigmatail,sigmaPEG,zhead,zPEG,bg):
        rho=[rhoPEG,1,rhotail,1,rhoPEG]
        sigma=[sigmaPEG,sigmahead,sigmatail,sigmahead,sigmaPEG]
        z=[-zPEG,-zhead,0,zhead,zPEG]
        I=np.zeros_like(x)
        for k1 in range(5):
            for k2 in range(5):
                I+=(R0+z[k1])*(R0+z[k2])*rho[k1]*rho[k2]*sigma[k1]*sigma[k2]*np.exp(-x**2*(sigma[k1]**2+sigma[k2]**2))*np.cos(x*(z[k1]-z[k2]))
        return A*I/x**2+bg

def VParabola(q,z0,h,w,zmin,zmax):
    return 4*np.pi*h*((zmax**3/3.*(1-z0**2/w**2)-zmax**5/(5*w**2)+z0*zmax**4/2/w**2)-
                      (zmin**3/3.*(1-z0**2/w**2)-zmin**5/(5*w**2)+z0*zmin**4/2/w**2))  

def FParabola(q,z0,h,w,zmin,zmax):
    term1=((w**2-z0**2)*q**2+4*z0*zmin*q**2-3*zmin**2*q**2+6)*np.sin(q*zmin)+((z0**2-w**2)*zmin*q**3+4*z0*q-2*zmin**2*z0*q**3-6*zmin*q+zmin**3*q**3)*np.cos(q*zmin)
    term2=((w**2-z0**2)*q**2+4*z0*zmax*q**2-3*zmax**2*q**2+6)*np.sin(q*zmax)+((z0**2-w**2)*zmax*q**3+4*z0*q-2*zmax**2*z0*q**3-6*zmax*q+zmax**3*q**3)*np.cos(q*zmax)
    
    return 4*np.pi*h/(q**5*w**2)*(term2-term1)

def Frhor(q,rmax,dr,rho):
    F=np.zeros_like(q)
    r=np.arange(0,rmax,dr)
    for i in range(len(q)):
        F[i]=np.trapz(r**2*rho*sinxdivx(r*q[i]),r)
    return 4*np.pi*F

class FFBilayerParabolicPEG(FitFunction):
    name="Bilayer from three Gaussians and parabolic PEG"
    formula="3 Gaussian Shells and two half parabolae"
    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
                   ('NR','Number of R points'),
                   ('rhotail','max SLD of tail region'),
                   ('rhoPEG','max SLD of PEG chains'),
                   ('sigmahead','HWHM of head regions'),
                   ('sigmatail','HWHM of tail region'),
                   ('wPEG','width of PEG region'),
                   ('zhead','barycenter of head regions, with respect to the tail position'),
                   ('zPEG','z0 of PEG regions, with respect to the tail position'),
                   ('bg','Constant background'),
                  ]    
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,R0,dR,NR,rhotail,rhoPEG,sigmahead,sigmatail,wPEG,zhead,zPEG,bg):
        r=np.linspace(0,R0+3*dR,NR)
        if len(r)==1 or dR==0:
            weight=np.ones_like(r)
        else:
            weight=np.exp(-(r-R0)**2/(2*dR**2))
        I=np.zeros_like(x)
        for i in range(len(r)):
            R=r[i]
            F=(rhotail*FGaussianShell(x,R,sigmatail)+
              1*FGaussianShell(x,R+zhead,sigmahead)+
              1*FGaussianShell(x,R-zhead,sigmahead)+
              FParabola(x,R-zPEG,rhoPEG,wPEG,R-zPEG,R)+
              FParabola(x,R+zPEG,rhoPEG,wPEG,R,R+zPEG))
            I+=F*F*weight[i]
        return A*I/weight.sum()+bg
    def rhoz(self,rhotail,rhoPEG,sigmahead,sigmatail,wPEG,zhead,zPEG):
        zmax=zPEG+wPEG
        z=np.linspace(-zmax,zmax,1000)
        rho=np.zeros_like(z)
        rho+=1*np.exp(-(z-zhead)**2/(2.*sigmahead**2))
        rho+=1*np.exp(-(z+zhead)**2/(2.*sigmahead**2))
        rho+=rhotail*np.exp(-(z)**2/(2.*sigmatail**2))
        rho[(z<=-zPEG)&(z>=-zmax)]=rhoPEG*(1-1./wPEG**2*(z[(z<=-zPEG)&(z>=-zmax)]+zPEG)**2)
        rho[(z>=zPEG)&(z<=zmax)]=rhoPEG*(1-1./wPEG**2*(z[(z>=zPEG)&(z<=zmax)]-zPEG)**2)
        return z,rho
        
        
class FFSSL(FitFunction):
    name="Sterically Stabilized Liposome"
    formula="Castorph et al. Biophys. J. 98(7) 1200-8"
    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
                   ('NR','Number of R points'),
                   ('rhohead','SLD of lipid head region'),
                   ('rhotail','SLD of lipid tail region'),
                   ('rhoc','SLD of PEG chains'),
                   ('thead','half thickness of head region'),
                   ('ttail','half thickness of tail region'),
                   ('Rgin','Rg of inner PEG corona'),
                   ('Rgout','Rg of outer PEG corona'),
                   ('ncin','Surface density of inner PEGs'),
                   ('ncout','Surface density of outer PEGs'),
                   ('bg','Constant background'),
                  ]
    defaultargs=[3.16192e-28,400,50,50,0.53238,-1.7503,1,9.34763,6.60621,10,10,7.09e-3,0.47e-3,0.0209624]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,R0,dR,NR,rhohead,rhotail,rhoc,thead,ttail,Rgin,Rgout,ncin,ncout,bg):
        r=np.linspace(R0-3*dR,R0+3*dR,NR)
        if len(r)==1 or dR==0:
            weight=np.ones_like(r)
        else:
            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2)
        I=np.zeros_like(x)
        # A Gauss-gombhejak effektiv vastagsaga (a Gauss fuggvenyt helyettesito
        # ugyanakkora amplitudoju es integralu lepcsofuggveny teljes szelessege)
        whead=thead*np.sqrt(2*np.pi)
        wtail=ttail*np.sqrt(2*np.pi)
       
        for i in range(len(r)):
            #dimenziok szamolasa
            R=r[i] # aktualis liposzomameret (legkulso sugar, PEG reteget leszamitva)
            # D=whead*2+wtail # A kettosreteg vastagsaga
            # A liposzomat alkoto harom Gauss-reteg sulyponti sugarai
            Rin=R-(whead*3./2+wtail) 
            Rout=R-whead/2
            Rtail=R-whead-wtail/2
            # magyarazat:
            #    Rout + whead/2 a kulso fejcsoportregio kulso szele, azaz a
            #        liposzoma meretet ez adja meg, ez egyenlo R-rel
            #    A szenlancregiot szimbolizalo Gauss profil kozepe a kulso fej-
            #        csoportregio kozepetol wtail/2 + whead/2 tavolsagra kell,
            #        hogy legyen, hogy az ekvivalens lepcsoprofilok pont erint-
            #        kezzenek -> Rtail adodik.
            #    A belso fejcsoport Gauss-a es a szenlanc Gauss-a az elozohoz
            #        hasonlo megfontolasok alapjan wtail/2+whead/2 tavolsagra
            #        vannak egymastol
            
            
            # A PEG retegek sulyponti sugarai, a liposzoma kulso es belso
            # feluletetol Rg tavolsagra (PEG lanc es a kettosreteg metszesenek
            # nagyjaboli kikuszobolese)
            #RPEGin=Rtail-(D/2+Rgin)
            RPEGin=Rin-wtail/2-Rgin
            #RPEGout=Rtail+(D/2+Rgout)             
            RPEGout=R+Rgout # ugyanaz, mint az elozo sor
            # A PEG molekulak darabszama az aktualis liposzoma kulso es belso
            # feluleten, fix feluleti darabszamsuruseget feltetelezve, azaz az
            # RPEG{in,out} sugaru gomb feluleten nc{in,out}*Felulet darab van.
            Nin=ncin*4*np.pi*RPEGin**2
            Nout=ncout*4*np.pi*RPEGout**2
            # Egyetlen belso PEG terfogata
            VolPEGin1=4*np.pi/3*Rgin**3
            # Egyetlen kulso PEG terfogata
            VolPEGout1=4*np.pi/3*Rgout**3
                        
            ### formafaktorok: F(q=0)=integral(deltarho) konvencioval, ahol deltarho
            ### a kozeghez (viz) kepesti fennmarado elektronsuruseg
            
            # A kettosreteg kulonbozo komponensei
            Fbin=rhohead*FGaussianShell(x,Rin,thead)
            Fbout=rhohead*FGaussianShell(x,Rout,thead)
            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
            #a kettosreteg teljes formafaktora
            Fb=Fbin+Fbout+Fbtail #osszeadva normalizalas nelkul
            #Egyetlen PEG lanc formafaktora a belso retegbol
            FPEGin1=rhoc*VolPEGin1*FGaussianChain(x,Rgin)
            #Egyetlen PEG lanc formafaktora a kulso retegbol
            FPEGout1=rhoc*VolPEGout1*FGaussianChain(x,Rgout)
            #Egyetlen belso PEG lanc oninterferencia-tagja:
            F2PEGin1=rhoc**2*VolPEGin1**2*F2GaussianChain(x,Rgin)
            #Egyetlen kulso PEG lanc oninterferencia-tagja:
            F2PEGout1=rhoc**2*VolPEGout1**2*F2GaussianChain(x,Rgout)
            #A kulso PEG lancok alkotta hej formafaktora mas entitasokkal valo interferencia szamolasara
            FPEGshellin=Nin*FPEGin1*sinxdivx(x*RPEGin)
            #A belso PEG lancok alkotta hej formafaktora mas entitasokkal valo interferencia szamolasara
            FPEGshellout=Nout*FPEGout1*sinxdivx(x*RPEGout)
            
            #rakjuk ossze a szort intenzitasat ennek a liposzomanak:
                
            thisI = (Fb**2 +  # a lipid kettosreteg oninterferenciaja
                  F2PEGin1*Nin + F2PEGout1*Nout + # a PEG lancok oninterferenciai
                  2*Fb*FPEGshellin+2*Fb*FPEGshellout + # a kettosreteg es a PEG hejak interferenciai
                  FPEGshellin*FPEGshellin*(Nin-1)/Nin + # a belso PEG reteg egyik PEG-jenek kolcsonhatasa a tobbivel
                  FPEGshellout*FPEGshellout*(Nout-1)/Nout + # a kulso PEG reteg egyik PEG-jenek kolcsonhatasa a tobbivel
                  FPEGshellout*FPEGshellin) #a kulso es a belso PEG hejak interferenciaja
                  
            #Mivel minden formafaktor-jellegu mennyiseg beepitve tartalmazta az
            # elektronsuruseget (a dimenzio elektronsuruseg*terfogat = elektron
            # darabszam), thisI is olyan dimenzioju (elektron darabszam**2)
                       
            I+=thisI*weight[i]
            
            print "RPEGin:",RPEGin
            print "RPEGout:",RPEGout
            print "Nin:",Nin
            print "Nout:",Nout
            print "Rhoc*VolPEGout1:",rhoc*VolPEGout1
        return A*I/weight.sum()+bg



class FFLorentzian(FitFunction):
    name="Lorentzian peak"
    formula="y(x)=A/(1+((x-x0)/sigma)^2)+B"
    argument_info=[('A','Scaling'),('x0','Center'),('sigma','HWHM'),('B','offset')]
    def __init__(self):
        FitFunction.__init__(self)
    def __call__(self,x,A,x0,sigma,B):
        return A/(1+((x-x0)/sigma)**2)+B

def Fsphere(q,R):
    return 4*np.pi/q**3*(np.sin(q*R)-q*R*np.cos(q*R))

def Fsphere_outer(q,R):
    qR=np.outer(q,R)
    q1=np.outer(q,np.ones_like(R))
    return 4*np.pi/q1**3*(np.sin(qR)-qR*np.cos(qR))

def F2cylinder_scalar(q,R,L,Nstep=1000):
    factor=16*(R*R*L*np.pi)**2
    lim1=(j1(q*R)/(q*R)*0.5)**2
    lim2=(0.5*np.sin(q*L*0.5)/(q*L))**2
    lim=max(lim1,lim2)
    x=np.random.rand(Nstep)
    y=np.random.rand(Nstep)*lim
    return factor*((y-(j1(q*R*np.sqrt(1-x**2))/(q*R*np.sqrt(1-x**2))*np.sin(q*L*x*0.5)/(q*L*x))**2)<0).sum()/Nstep

class FFCaelyx(FitFunction):
    name='Caelyx: PEGylated lipid bilayer from step functions and a cylinder inside'
    formula="CaelyxInt()"
    argument_info=[('rhohead',''),
                   ('rhotail',''),
                   ('rhoPEG',''),
                   ('rhodox',''),
                   ('R',''),
                   ('dR',''),
                   ('whead',''),
                   ('wtail',''),
                   ('wPEG',''),
                   ('Ldox',''),
                   ('Rdox',''),
                   ('Niter',''),
                  ]
    def __call__(self,x,rhohead,rhotail,rhoPEg,rhodox,R,dR,whead,wtail,wPEG,Ldox,Rdox,Niter):
        return _fitfunction.IntCaelyx(x,rhohead,rhotail,rhoPEg,rhodox,R,dR,whead,wtail,wPEG,Ldox,Rdox,long(Niter))
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
        
class TransformGuinier(Transform):
    _factory_arguments=[(0,),(1,),(2,)]
    name='Guinier'
    def __init__(self,qpower=0):
        self._qpower=qpower
        if qpower==0:
            self.name='Guinier (ln I vs. q)'
        elif qpower==1:
            self.name='Guinier cross-section (ln Iq vs. q)'
        elif qpower==2:
            self.name='Guinier thickness (ln Iq^2 vs. q)'
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        d['x']=np.power(x,2)
        d['y']=np.log(y*np.power(x,self._qpower))
        if dy is not None:
            d['dy']=np.absolute(dy/y)
        if dx is not None:
            d['dx']=2*np.absolute(x)*dx
        return d

class TransformLogLog(Transform):
    name=''
    _factory_arguments=[(True,True),(True,False),(False,True),(False,False)]
    def __init__(self,xlog=True,ylog=True):
        self._xlog=xlog
        self._ylog=ylog
        if xlog and ylog:
            self.name='Double logarithmic'
        elif xlog:
            self.name='Logarithmic x'
        elif ylog:
            self.name='Logarithmic y'
        else:
            self.name='Linear'
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        if self._xlog:
            d['x']=np.log(x)
            if dx is not None:
                d['dx']=np.absolute(dx/x)
        else:
            d['x']=np.array(x) # make a copy!
            if dx is not None:
                d['dx']=np.array(dx) # make a copy!
        if self._ylog:
            d['y']=np.log(y)
            if dy is not None:
                d['dy']=np.absolute(dy/y)
        else:
            d['y']=np.array(y) # make a copy!
            if dy is not None:
                d['dy']=np.array(dy) # make a copy!
        return d


class TransformPorod(Transform):
    name='Porod'
    _factory_arguments=[(4,),(3,)]
    def __init__(self,exponent=4):
        self._exponent=exponent
        self.name='Porod (q^%s)'%exponent
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        d['x']=np.power(x,self._exponent)
        d['y']=np.power(x,self._exponent)*y
        if dy is not None:
            d['dy']=np.power(x,self._exponent)*dy
        if dx is not None:
            d['dx']=np.power(x,self._exponent-1)*(self._exponent)*dx
        return d

class TransformKratky(Transform):
    name='Porod'
    _factory_arguments=[(2,)]
    def __init__(self,exponent=2):
        self._exponent=exponent
        self.name='Kratky (Iq^%s vs. q)'%exponent
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        d['x']=np.power(x,self._exponent)
        d['y']=np.power(x,self._exponent)*y
        if dy is not None:
            d['dy']=np.power(x,self._exponent)*dy
        if dx is not None:
            d['dx']=np.power(x,self._exponent-1)*(self._exponent)*dx
        return d


class TransformShullRoess(Transform):
    name='Shull-Roess'
    _factory_arguments=None
    def __init__(self,r0):
        self._r0=r0
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        d['x']=np.log(np.power(x,2)+3/self._r0**2)
        d['y']=np.log(y)
        if dy is not None:
            d['dy']=np.absolute(dy/y)
        if dx is not None:
            d['dx']=2*x*dx/(np.power(x,2)+3/self._r0**2)
        return d

class TransformZimm(Transform):
    name='Zimm'
    def __init__(self):
        pass
    def do_transform(self,x,y,dy=None,dx=None,**kwargs):
        d=kwargs
        d['x']=np.power(x,2)
        d['y']=1/y
        if dy is not None:
            d['dy']=dy/y
        if dx is not None:
            d['dx']=2*np.absolute(x)*dx
        return d

def Factory(superclass=FitFunction):
    productlist=[]
    for e in superclass.__subclasses__():
        if hasattr(e,'_factory_arguments'):
            # if this requires special treatment
            if e._factory_arguments is None:
                #skip this.
                continue
            for x in e._factory_arguments:
                productlist.append(e(*x))
        else:
            productlist.append(e())
    return sorted(productlist,key=lambda x:x.name)
