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

#def Frhor(q,rmax,dr,rho):
#    F=np.zeros_like(q)
#    r=np.arange(0,rmax,dr)
#    for i in range(len(q)):
#        F[i]=np.trapz(r**2*rho*sinxdivx(r*q[i]),r)
#    return 4*np.pi*F







#def F2cylinder_scalar(q,R,L,Nstep=1000):
#    factor=16*(R*R*L*np.pi)**2
#    lim1=(j1(q*R)/(q*R)*0.5)**2
#    lim2=(0.5*np.sin(q*L*0.5)/(q*L))**2
#    lim=max(lim1,lim2)
#    x=np.random.rand(Nstep)
#    y=np.random.rand(Nstep)*lim
#    return factor*((y-(j1(q*R*np.sqrt(1-x**2))/(q*R*np.sqrt(1-x**2))*np.sin(q*L*x*0.5)/(q*L*x))**2)<0).sum()/Nstep

       
        

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
