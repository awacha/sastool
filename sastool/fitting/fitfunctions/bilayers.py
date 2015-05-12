import numpy as np
from .sasbasic import Fsphere
from scipy.special import sinc

__all__ = []

##--------------- helper functions
#def Fsphere_outer(q,R):
#    qR=np.outer(q,R)
#    q1=np.outer(q,np.ones_like(R))
#    return 4*np.pi/q1**3*(np.sin(qR)-qR*np.cos(qR))
#
#def F2GaussianChain(q,Rg):
#    """Squared form factor (self interference term) of a Gaussian chain
#    
#    Inputs:
#        q: q values
#        Rg: radius of gyration
#        
#    Outputs:
#        the squared form factor, 2*(exp(-x)-1+x)/x^2, i.e. the Debye function,
#            where x=(q*Rg)^2. The convention F(q=0)=1 is used.
#    """
#    x=(q*Rg)**2
#    if np.isscalar(x):
#        if x==0:
#            return 1
#        else:
#            return 2*(np.exp(-x)-1+x)/x**2
#    y=np.ones_like(x)
#    idxok=x!=0
#    y[idxok]=2*(np.exp(-x[idxok])-1+x[idxok])/(x[idxok])**2
#    return y
#
#
#def FGaussianShell(q,R,sigma):
#    """Scattering factor of a spherical shell with a radial Gaussian el.density profile
#    
#    Inputs:
#        q: q values
#        R: radius
#        sigma: HWHM of the radial Gauss profile
#        
#    Outputs:
#        the form factor (same format as q), with the convention F(q=0)=V
#        
#    Notes:
#        Gradzielski & al (J. Phys. Chem. 99 pp 13232-8)
#    """
#    return 4*np.pi*np.sqrt(2*np.pi)*sigma/q*np.exp(-(q*sigma)**2/2)*(R*np.sin(q*R)+q*sigma**2*np.cos(q*R))
#
#def VGaussianShell(R,t):
#    """Volume of a Gaussian Shell
#    """
#    return 4*np.pi*np.sqrt(2*np.pi)*t*(R**2+t**2)
#
#def FGaussianChain(q,Rg):
#    """Form factor (sqrt!!) of a Gaussian chain
#    
#    Inputs:
#        q: q values
#        Rg: radius of gyration
#        
#    Outputs:
#        the form factor, 1-exp(-(q*Rg)**2)/(q*Rg), F(q=0)=1
#    """
#    x=(q*Rg)**2
#    if np.isscalar(x):
#        if x==0:
#            return 1
#        else:
#            return (1-np.exp(-x))/x
#    y=np.ones_like(x)
#    idxok=x!=0
#    y[idxok]=(1-np.exp(-x[idxok]))/(x[idxok])
#    return y
#
#
#        
#
#def VParabola(q,z0,h,w,zmin,zmax):
#    return 4*np.pi*h*((zmax**3/3.*(1-z0**2/w**2)-zmax**5/(5*w**2)+z0*zmax**4/2/w**2)-
#                      (zmin**3/3.*(1-z0**2/w**2)-zmin**5/(5*w**2)+z0*zmin**4/2/w**2))  
#
#def FParabola(q,z0,h,w,zmin,zmax):
#    term1=((w**2-z0**2)*q**2+4*z0*zmin*q**2-3*zmin**2*q**2+6)*np.sin(q*zmin)+((z0**2-w**2)*zmin*q**3+4*z0*q-2*zmin**2*z0*q**3-6*zmin*q+zmin**3*q**3)*np.cos(q*zmin)
#    term2=((w**2-z0**2)*q**2+4*z0*zmax*q**2-3*zmax**2*q**2+6)*np.sin(q*zmax)+((z0**2-w**2)*zmax*q**3+4*z0*q-2*zmax**2*z0*q**3-6*zmax*q+zmax**3*q**3)*np.cos(q*zmax)
#    
#    return 4*np.pi*h/(q**5*w**2)*(term2-term1)
#
#
#
#
#
##--------------- fit functions (subclasses of FitFunction)
#
#class FGaussianBilayerExact(FitFunction):
#    name="PEG-decorated bilayer, Gauss profiles"
#    formula=""
#    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
#                   ('NR','Number of R points'),
#                   ('rhohead','SLD of lipid head region'),
#                   ('rhotail','SLD of lipid tail region'),
#                   ('rhoPEG','SLD of PEG chain region'),
#                   ('zhead','distance of the head groups from the center of the bilayer'),
#                   ('zPEG','distance of the PEG chains from the center of the bilayer'),
#                   ('thead','HWHM of the head group profile'),
#                   ('ttail','HWHM of the head group profile'),
#                   ('tPEG','HWHM of the head group profile'),
#                   ('bg','Constant background')
#                  ]
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,A,R0,dR,NR,rhohead,rhotail,rhoPEG,zhead,zPEG,thead,ttail,tPEG,bg):
#        r=np.linspace(0,R0+3*dR,NR)
#        if len(r)==1 or dR==0:
#            weight=np.ones_like(r)
#        else:
#            weight=np.exp(-(r-R0)**2/(2*dR**2))
#        I=np.zeros_like(x)
#        for i in range(len(r)):
#            #dimenziok szamolasa
#            R=r[i] # aktualis liposzomameret (legkulso sugar, PEG reteget leszamitva)
#            # D=whead*2+wtail # A kettosreteg vastagsaga
#            # A liposzomat alkoto harom Gauss-reteg sulyponti sugarai
#            Rin=R-zhead 
#            Rout=R+zhead
#            Rtail=R
#            RPEGout=R+zPEG
#            RPEGin=R-zPEG
#            Fbin=rhohead*FGaussianShell(x,Rin,thead)
#            Fbout=rhohead*FGaussianShell(x,Rout,thead)
#            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
#            FbPEGin=rhoPEG*FGaussianShell(x,RPEGin,tPEG)
#            FbPEGout=rhoPEG*FGaussianShell(x,RPEGout,tPEG)
#            thisI=(Fbin+Fbout+Fbtail+FbPEGin+FbPEGout)**2*weight[i]
#            I+=thisI
#        return I/weight.sum()*A+bg
#
#
#
#class FFBilayer3Gauss(FitFunction):
#    name="Bilayer form factor from 3 Gaussians"
#    formula="y(x)=Bilayer3Gauss(x,R0,dR,NR,thead,ttail,rhohead,rhotail)"
#    argument_info=[('R0','Mean radius of vesicles'),('dR','HWHM of R'),('NR','R discretization'),
#                   ('thead','Thickness of head groups'),
#                   ('ttail','Length of tail region'),('rhohead','SLD of head groups'),
#                   ('rhotail','SLD of tail'),
#                   ('bg','Constant background')]
#    defaultargs=[200,50,100,3,3.5,380,304,0]
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,R0,dR,NR,thead,ttail,rhohead,rhotail,bg):
#        whead=thead*np.sqrt(2*np.pi)
#        wtail=ttail*np.sqrt(2*np.pi)
#        r=np.linspace(R0-dR*3,R0+dR*3,NR);
#        if len(r)==1 or dR==0:
#            weight=np.ones_like(r)
#        else:
#            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2);
#        I=np.zeros_like(x)
#        for i in range(len(r)):
#            R=r[i]
#            Rout=R-whead/2
#            Rtail=R-(whead+wtail/2)
#            Rin=R-(whead+wtail+whead/2)
#            
#            Fbin=rhohead*FGaussianShell(x,Rin,thead)
#            Fbout=rhohead*FGaussianShell(x,Rout,thead)
#            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
#            I+=(Fbin+Fbout+Fbtail)**2*weight[i]
#        return I/weight.sum()+bg;
#
#class FFBilayer3Step(FitFunction):
#    name="Bilayer form factor from 3 Step functions"
#    formula="y(x)=Bilayer3Step(x,R0,dR,NR,whead,wtail,rhohead,rhotail)"
#    argument_info=[('R0','Mean radius of vesicles'),('dR','HWHM of R'),('NR','R discretization'),
#                   ('whead','Thickness of head groups'),
#                   ('wtail','Length of tail region'),('rhohead','SLD of head groups'),
#                   ('rhotail','SLD of tail'),
#                   ('bg','Constant background')]
#    defaultargs=[200,50,100,3,3.5,380,304,0]
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,R0,dR,NR,whead,wtail,rhohead,rhotail,bg):
#        r=np.linspace(R0-dR*3,R0+dR*3,NR);
#        if len(r)==1 or dR==0:
#            weight=np.ones_like(r)
#        else:
#            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2);
#        I=np.zeros_like(x)
#        for i in range(len(r)):
#            R=r[i]
#            Rout=R-whead/2
#            Rtail=R-whead-wtail/2
#            Rin=R-whead-wtail-whead/2
#            Fbin=Fsphere(x,Rin+whead/2)-Fsphere(x,Rin-whead/2)
#            Fbout=Fsphere(x,Rout+whead/2)-Fsphere(x,Rout-whead/2)
#            Fbtail=Fsphere(x,Rtail+wtail/2)-Fsphere(x,Rtail-wtail/2)
#            I+=(rhohead/whead*(Fbin+Fbout)+rhotail/wtail*Fbtail)**2*weight[i]
#        return I/weight.sum()+bg;
#        
#class FFMicelleLogNorm(FitFunction):
#    name="Micelle scattered intensity from step functions, with log-normal size distribution"
#    formula="y(x)=Micelle(x,Factor,rhohead,rhotail,rhoPEG,whead,dwhead,wtail,dwtail,wPEG,dwPEG,N,bg)"
#    argument_info=[('Factor','Scaling factor'),
#                   ('Rout','Mean value of the outer radius'),
#                   ('sigma','Sigma parameter of the log-normal distribution of radii'),
#                   ('rhohead','SLD of head groups'),
#                   ('rhotail','SLD of hydrocarbon chains'),
#                   ('rhoPEG','SLD of PEG chains'),
#                   ('whead','thickness of head groups'),
#                   ('shead','stretching weight of the head groups'),
#                   ('wtail','length of the hydrocarbon chain region'),
#                   ('stail','stretching weight of the tail'),
#                   ('wPEG','length of the PEG chains'),
#                   ('sPEG','stretching weight of the PEG chains'),
#                   ('N','Number of MC steps'),
#                   ('bg','Constant background')]
#    defaultargs=[1, 5, 0.001, 0.443-0.333, 0.302-0.333, 0.3385-0.333, 0.5, 
#                 1, 1.6, 1, 4.6, 1, 100, 0]
#    def __call__(self, x, Factor, Rout, sigma, rhohead, rhotail, rhoPEG, whead, 
#                 shead, wtail, stail, wPEG, sPEG, N, bg):
#        # N samples from a log-normal distribution
#        R=np.random.lognormal(np.log(Rout),sigma,N)
#        #normalize the stretching weights
#        sums=float(shead*whead+stail*wtail+sPEG*wPEG)
#        shead=shead*whead/sums
#        stail=stail*wtail/sums
#        sPEG=sPEG*wPEG/sums
#        #divide the difference between R and (whead+wtail+wPEG) among the three
#        # leaflets with respect to the stretching weights
#        residualR=R-(whead+wtail+wPEG)
#        whead=whead+residualR*shead
#        wPEG=wPEG+residualR*sPEG
#        #wtail=wtail+residualR*stail # no need for this quantity later
#        # PEG layer, as a hollow sphere
#        F=rhoPEG*(Fsphere_outer(x,R)-Fsphere_outer(x,R-wPEG))
#        # head groups, hollow sphere
#        F+=rhohead*(Fsphere_outer(x,R-wPEG)-Fsphere_outer(x,R-wPEG-whead))
#        # tail groups, full sphere
#        F+=rhotail*(Fsphere_outer(x,R-whead-wPEG))
#        # intensity
#        return Factor*(F.sum(1)/float(N))**2+bg
#
#
#class FFBilayerParabolicPEG(FitFunction):
#    name="Bilayer from three Gaussians and parabolic PEG"
#    formula="3 Gaussian Shells and two half parabolae"
#    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
#                   ('NR','Number of R points'),
#                   ('rhotail','max SLD of tail region'),
#                   ('rhoPEG','max SLD of PEG chains'),
#                   ('sigmahead','HWHM of head regions'),
#                   ('sigmatail','HWHM of tail region'),
#                   ('wPEG','width of PEG region'),
#                   ('zhead','barycenter of head regions, with respect to the tail position'),
#                   ('zPEG','z0 of PEG regions, with respect to the tail position'),
#                   ('bg','Constant background'),
#                  ]    
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,A,R0,dR,NR,rhotail,rhoPEG,sigmahead,sigmatail,wPEG,zhead,zPEG,bg):
#        r=np.linspace(0,R0+3*dR,NR)
#        if len(r)==1 or dR==0:
#            weight=np.ones_like(r)
#        else:
#            weight=np.exp(-(r-R0)**2/(2*dR**2))
#        I=np.zeros_like(x)
#        for i in range(len(r)):
#            R=r[i]
#            F=(rhotail*FGaussianShell(x,R,sigmatail)+
#              1*FGaussianShell(x,R+zhead,sigmahead)+
#              1*FGaussianShell(x,R-zhead,sigmahead)+
#              FParabola(x,R-zPEG,rhoPEG,wPEG,R-zPEG,R)+
#              FParabola(x,R+zPEG,rhoPEG,wPEG,R,R+zPEG))
#            I+=F*F*weight[i]
#        return A*I/weight.sum()+bg
#    def rhoz(self,rhotail,rhoPEG,sigmahead,sigmatail,wPEG,zhead,zPEG):
#        zmax=zPEG+wPEG
#        z=np.linspace(-zmax,zmax,1000)
#        rho=np.zeros_like(z)
#        rho+=1*np.exp(-(z-zhead)**2/(2.*sigmahead**2))
#        rho+=1*np.exp(-(z+zhead)**2/(2.*sigmahead**2))
#        rho+=rhotail*np.exp(-(z)**2/(2.*sigmatail**2))
#        rho[(z<=-zPEG)&(z>=-zmax)]=rhoPEG*(1-1./wPEG**2*(z[(z<=-zPEG)&(z>=-zmax)]+zPEG)**2)
#        rho[(z>=zPEG)&(z<=zmax)]=rhoPEG*(1-1./wPEG**2*(z[(z>=zPEG)&(z<=zmax)]-zPEG)**2)
#        return z,rho
#        
#        
#class FFSSL(FitFunction):
#    name="Sterically Stabilized Liposome"
#    formula="Castorph et al. Biophys. J. 98(7) 1200-8"
#    argument_info=[('A','Scaling'),('R0','Radius'),('dR','HWHM of Radius'),
#                   ('NR','Number of R points'),
#                   ('rhohead','SLD of lipid head region'),
#                   ('rhotail','SLD of lipid tail region'),
#                   ('rhoc','SLD of PEG chains'),
#                   ('thead','half thickness of head region'),
#                   ('ttail','half thickness of tail region'),
#                   ('Rgin','Rg of inner PEG corona'),
#                   ('Rgout','Rg of outer PEG corona'),
#                   ('ncin','Surface density of inner PEGs'),
#                   ('ncout','Surface density of outer PEGs'),
#                   ('bg','Constant background'),
#                  ]
#    defaultargs=[3.16192e-28,400,50,50,0.53238,-1.7503,1,9.34763,6.60621,10,10,7.09e-3,0.47e-3,0.0209624]
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,A,R0,dR,NR,rhohead,rhotail,rhoc,thead,ttail,Rgin,Rgout,ncin,ncout,bg):
#        r=np.linspace(R0-3*dR,R0+3*dR,NR)
#        if len(r)==1 or dR==0:
#            weight=np.ones_like(r)
#        else:
#            weight=np.exp(-(r-R0)**2/(2*dR**2))/np.sqrt(2*np.pi*dR**2)
#        I=np.zeros_like(x)
#        # A Gauss-gombhejak effektiv vastagsaga (a Gauss fuggvenyt helyettesito
#        # ugyanakkora amplitudoju es integralu lepcsofuggveny teljes szelessege)
#        whead=thead*np.sqrt(2*np.pi)
#        wtail=ttail*np.sqrt(2*np.pi)
#       
#        for i in range(len(r)):
#            #dimenziok szamolasa
#            R=r[i] # aktualis liposzomameret (legkulso sugar, PEG reteget leszamitva)
#            # D=whead*2+wtail # A kettosreteg vastagsaga
#            # A liposzomat alkoto harom Gauss-reteg sulyponti sugarai
#            Rin=R-(whead*3./2+wtail) 
#            Rout=R-whead/2
#            Rtail=R-whead-wtail/2
#            # magyarazat:
#            #    Rout + whead/2 a kulso fejcsoportregio kulso szele, azaz a
#            #        liposzoma meretet ez adja meg, ez egyenlo R-rel
#            #    A szenlancregiot szimbolizalo Gauss profil kozepe a kulso fej-
#            #        csoportregio kozepetol wtail/2 + whead/2 tavolsagra kell,
#            #        hogy legyen, hogy az ekvivalens lepcsoprofilok pont erint-
#            #        kezzenek -> Rtail adodik.
#            #    A belso fejcsoport Gauss-a es a szenlanc Gauss-a az elozohoz
#            #        hasonlo megfontolasok alapjan wtail/2+whead/2 tavolsagra
#            #        vannak egymastol
#            
#            
#            # A PEG retegek sulyponti sugarai, a liposzoma kulso es belso
#            # feluletetol Rg tavolsagra (PEG lanc es a kettosreteg metszesenek
#            # nagyjaboli kikuszobolese)
#            #RPEGin=Rtail-(D/2+Rgin)
#            RPEGin=Rin-wtail/2-Rgin
#            #RPEGout=Rtail+(D/2+Rgout)             
#            RPEGout=R+Rgout # ugyanaz, mint az elozo sor
#            # A PEG molekulak darabszama az aktualis liposzoma kulso es belso
#            # feluleten, fix feluleti darabszamsuruseget feltetelezve, azaz az
#            # RPEG{in,out} sugaru gomb feluleten nc{in,out}*Felulet darab van.
#            Nin=ncin*4*np.pi*RPEGin**2
#            Nout=ncout*4*np.pi*RPEGout**2
#            # Egyetlen belso PEG terfogata
#            VolPEGin1=4*np.pi/3*Rgin**3
#            # Egyetlen kulso PEG terfogata
#            VolPEGout1=4*np.pi/3*Rgout**3
#                        
#            ### formafaktorok: F(q=0)=integral(deltarho) konvencioval, ahol deltarho
#            ### a kozeghez (viz) kepesti fennmarado elektronsuruseg
#            
#            # A kettosreteg kulonbozo komponensei
#            Fbin=rhohead*FGaussianShell(x,Rin,thead)
#            Fbout=rhohead*FGaussianShell(x,Rout,thead)
#            Fbtail=rhotail*FGaussianShell(x,Rtail,ttail)
#            #a kettosreteg teljes formafaktora
#            Fb=Fbin+Fbout+Fbtail #osszeadva normalizalas nelkul
#            #Egyetlen PEG lanc formafaktora a belso retegbol
#            FPEGin1=rhoc*VolPEGin1*FGaussianChain(x,Rgin)
#            #Egyetlen PEG lanc formafaktora a kulso retegbol
#            FPEGout1=rhoc*VolPEGout1*FGaussianChain(x,Rgout)
#            #Egyetlen belso PEG lanc oninterferencia-tagja:
#            F2PEGin1=rhoc**2*VolPEGin1**2*F2GaussianChain(x,Rgin)
#            #Egyetlen kulso PEG lanc oninterferencia-tagja:
#            F2PEGout1=rhoc**2*VolPEGout1**2*F2GaussianChain(x,Rgout)
#            #A kulso PEG lancok alkotta hej formafaktora mas entitasokkal valo interferencia szamolasara
#            FPEGshellin=Nin*FPEGin1*sinc(x*RPEGin/np.pi)
#            #A belso PEG lancok alkotta hej formafaktora mas entitasokkal valo interferencia szamolasara
#            FPEGshellout=Nout*FPEGout1*sinc(x*RPEGout/np.pi)
#            
#            #rakjuk ossze a szort intenzitasat ennek a liposzomanak:
#                
#            thisI = (Fb**2 +  # a lipid kettosreteg oninterferenciaja
#                  F2PEGin1*Nin + F2PEGout1*Nout + # a PEG lancok oninterferenciai
#                  2*Fb*FPEGshellin+2*Fb*FPEGshellout + # a kettosreteg es a PEG hejak interferenciai
#                  FPEGshellin*FPEGshellin*(Nin-1)/Nin + # a belso PEG reteg egyik PEG-jenek kolcsonhatasa a tobbivel
#                  FPEGshellout*FPEGshellout*(Nout-1)/Nout + # a kulso PEG reteg egyik PEG-jenek kolcsonhatasa a tobbivel
#                  FPEGshellout*FPEGshellin) #a kulso es a belso PEG hejak interferenciaja
#                  
#            #Mivel minden formafaktor-jellegu mennyiseg beepitve tartalmazta az
#            # elektronsuruseget (a dimenzio elektronsuruseg*terfogat = elektron
#            # darabszam), thisI is olyan dimenzioju (elektron darabszam**2)
#                       
#            I+=thisI*weight[i]
#            
#            print "RPEGin:",RPEGin
#            print "RPEGout:",RPEGout
#            print "Nin:",Nin
#            print "Nout:",Nout
#            print "Rhoc*VolPEGout1:",rhoc*VolPEGout1
#        return A*I/weight.sum()+bg
#
#class FGaussianBilayerPEG(FitFunction):
#    name="Gaussian Bilayer with PEG"
#    formula="Brzustowicz & Brunger, J. Appl. Cryst. 38 pp 126-31"
#    argument_info=[('A','Scaling'),
#                   ('R0','Mean radius'),
#                   ('rhotail','SLD of CH region'),
#                   ('rhoPEG','SLD of PEG chains'),
#                   ('sigmahead','SLD of head region'),
#                   ('sigmatail','SLD of tail region'),
#                   ('sigmaPEG','HWHM of PEG chains'),
#                   ('zhead','position of head region'),
#                   ('zPEG','position of the PEG chains'),
#                   ('bg','Constant background')]
#    def __init__(self):
#        FitFunction.__init__(self)
#    def __call__(self,x,A,R0,rhotail,rhoPEG,sigmahead,sigmatail,sigmaPEG,zhead,zPEG,bg):
#        rho=[rhoPEG,1,rhotail,1,rhoPEG]
#        sigma=[sigmaPEG,sigmahead,sigmatail,sigmahead,sigmaPEG]
#        z=[-zPEG,-zhead,0,zhead,zPEG]
#        I=np.zeros_like(x)
#        for k1 in range(5):
#            for k2 in range(5):
#                I+=(R0+z[k1])*(R0+z[k2])*rho[k1]*rho[k2]*sigma[k1]*sigma[k2]*np.exp(-x**2*(sigma[k1]**2+sigma[k2]**2))*np.cos(x*(z[k1]-z[k2]))
#        return A*I/x**2+bg
