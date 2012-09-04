import numpy as np

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
