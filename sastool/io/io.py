'Input-output routines'
import gzip
import zipfile
import string
import scipy.io
import warnings
import datetime
import numpy as np
import os

from sastool.misc import energycalibration, normalize_listargument

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

try:
    import Image
except ImportError:
    warnings.warn('Cannot import module Image (Python Imaging Library). Only Pilatus300k and Gabriel images can be loaded (Pilatus100k, 1M etc. NOT).')

def read2dintfile(fsns,dirs=[],norm=True,quiet=False):
    """Read corrected intensity and error matrices
    
    Input:
        fsns: one or more fsn-s in a list
        dirs: list of directories to try
        norm: True if int2dnorm*.mat file is to be loaded, False if
            int2darb*.mat is preferred. You can even supply the file prefix
            itself.
        quiet: True if no warning messages should be printed
        
    Output:
        a list of 2d intensity matrices
        a list of error matrices
        a list of param dictionaries
        dirs [optional]: a list of directories to try
    
    Note:
        It tries to load int2dnorm<FSN>.mat. If it does not succeed,
        it tries int2dnorm<FSN>.dat and err2dnorm<FSN>.dat. If these do not
        exist, int2dnorm<FSN>.dat.zip and err2dnorm<FSN>.dat.zip is tried. If
        still no luck, int2dnorm<FSN>.dat.gz and err2dnorm<FSN>.dat.gz is
        opened. If this fails as well, the given FSN is skipped. If no files
        have been loaded, empty lists are returned.
        If the shape of the loaded error matrix is not equal to that of the
        intensity, the error matrix is overridden with a zero matrix.
    """
    def read2dfromstream(stream):
        """Read 2d ascii data from stream.
        It uses only stream.readlines()
        Watch out, this is extremely slow!
        """
        lines=stream.readlines()
        M=len(lines)
        N=len(lines[0].split())
        data=np.zeros((M,N),order='F')
        for l in range(len(lines)):
            data[l]=[float(x) for x in lines[l].split()];
        del lines
        return data
    def read2dascii(filename):
        """Read 2d data from an ascii file
        If filename is not found, filename.zip is tried.
        If that is not found, filename.gz is tried.
        If that is not found either, return None.
        """
        try:
            fid=open(filename,'r')
            data=read2dfromstream(fid)
            fid.close()
        except IOError:
            try:
                z=zipfile.ZipFile(filename+'.zip','r')
                fid=z.open(filename)
                data=read2dfromstream(fid)
                fid.close()
                z.close()
            except KeyError:
                z.close()
            except IOError:
                try:
                    z=gzip.GzipFile(filename+'.gz','r')
                    data=read2dfromstream(z)
                    z.close()
                except IOError:
#                    print 'Cannot find file %s (also tried .zip and .gz)' % filename
                    return None
        return data
    # the core of read2dintfile
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if np.isscalar(fsns):
        fsns=[fsns]
    int2d=[]
    err2d=[]
    params=[]
    for fsn in fsns: # this also works if len(fsns)==1
        filefound=False
        for d in dirs:
            try: # first try to load the npz file. This is the most effective way.
                if isinstance(norm,str):
                    fileprefixnorm=norm
                elif norm:
                    fileprefixnorm='int2dnorm'
                else:
                    fileprefixnorm='int2darb'
                tmp0=np.load(os.path.join(d,'%s%d.npz' % (fileprefixnorm,fsn)))
                tmp=tmp0['Intensity']
                tmp1=tmp0['Error']
            except IOError:
                try: # first try to load the mat file. This is the second effective way.
                    tmp0=scipy.io.loadmat(os.path.join(d,'%s%d.mat' % (fileprefixnorm,fsn)))
                    tmp=tmp0['Intensity']
                    tmp1=tmp0['Error']
                except IOError: # if mat file is not found, try the ascii files
                    if isinstance(norm,str):
                        warnings.warn(SyntaxWarning('Loading 2D ascii files when parameter <norm> for read2dintfile() is a string.'))
                        continue # try from another directory
                    if norm:
    #                    print 'Cannot find file int2dnorm%d.mat: trying to read int2dnorm%d.dat(.gz|.zip) and err2dnorm%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2dnorm%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2dnorm%d.dat' % (d,os.sep,fsn));
                    else:
    #                    print 'Cannot find file int2darb%d.mat: trying to read int2darb%d.dat(.gz|.zip) and err2darb%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2darb%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2darb%d.dat' % (d,os.sep,fsn));
                except TypeError: # if mat file was found but scipy.io.loadmat was unable to read it
                    if not quiet:
                        print "Malformed MAT file! Skipping."
                    continue # try from another directory
            if (tmp is not None) and (tmp1 is not None): # if all of int,err and log is read successfully
                filefound=True
#                print 'Files corresponding to fsn %d were found.' % fsn
                break # file was found, do not try to load it again from another directory
        if filefound:
            tmp2=readlogfile(fsn,dirs=dirs)[0]
            if len(tmp2)>0:
                int2d.append(tmp)
                if tmp1.shape!=tmp.shape:    # test if the shapes of intensity and error matrices are the same. If not, let the error matrix be a zero matrix of the same size as the intensity.
                    tmp1=np.zeros(tmp.shape)
                err2d.append(tmp1)
                params.append(tmp2)                
        if not filefound and not quiet:
            print "read2dintfile: Cannot find file(s ) for FSN %d" % fsn
    return int2d,err2d,params # return the lists

def write2dintfile(A,Aerr,params,norm=True,filetype='npz'):
    """Save the intensity and error matrices to int2dnorm<FSN>.mat
    
    Inputs:
        A: the intensity matrix
        Aerr: the error matrix (can be None, if no error matrix is to be saved)
        params: the parameter dictionary
        norm: if int2dnorm files are to be saved. If it is false, int2darb files
            are saved (arb = arbitrary units, ie. not absolute intensity). If a string,
            save it to <norm>%d.<filetype>.
        filetype: 'npz' or 'mat'
    int2dnorm<FSN>.[mat or npz] is written. The parameter structure is not
        saved, since it should be saved already in intnorm<FSN>.log
    """
    if Aerr is None:
        Aerr=np.zeros((1,1))
    if isinstance(norm,str):
        fileprefix='%s%d' % (norm,params['FSN'])
    elif norm:
        fileprefix='int2dnorm%d' % params['FSN']
    else:
        fileprefix='int2darb%d' % params['FSN']
    if filetype.upper() in ['NPZ','NPY','NUMPY']:
        np.savez('%s.npz' % fileprefix, Intensity=A,Error=Aerr)
    elif filetype.upper() in ['MAT','MATLAB']:
        scipy.io.savemat('%s.mat' % fileprefix,{'Intensity':A,'Error':Aerr});
    else:
        raise ValueError,"Unknown file type: %s" % repr(filetype)

def readintfile(filename,dirs=[],quiet=False):
    """Read intfiles.

    Input:
        filename: the file name, eg. intnorm123.dat. If it ends with ".mat" or
            ".npy", reads it as a binary file.
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed

    Output:
        A dictionary with 'q' holding the values for the momentum transfer,
            'Intensity' being the intensity vector and 'Error' has the error
            values. These three fields are numpy ndarrays. An empty dict
            is returned if file is not found.
    """
    fields=['q','Intensity','Error','Area','qavg','qstd']
    
    if not (isinstance(dirs,list) or isinstance(dirs,tuple) or isinstance(dirs,set)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    ret={}
    for d in dirs:
        try:
            if d=='.':
                fname=filename
            else:
                fname= "%s%s%s" % (d,os.sep,filename)
            if fname.upper().endswith('.MAT'):
                #try to open MAT file
                f=scipy.io.loadmat(fname) #raises IOError if file not found
                try:
                    firstkey=[k for k in f.keys() if not (k.startswith('__') and k.endswith('__'))][0]
                except IndexError:
                    raise RuntimeError('MAT file is empty!')
                ret=dict([(n,f[firstkey][:,i]) for n,i in zip(fields,range(f[firstkey].shape[1]))]);
            elif fname.upper().endswith('.NPY'):
                #try to open npy file
                f=np.load(fname);
                if f.dtype.names is not None: #we are dealing with a structured array
                    ret=dict([(n,f[n]) for n in f.dtype.names])
                else:
                    ret=dict([(n,f[:,i]) for n,i in zip(fields,range(f.shape[1]))]);
            else: #assume text file
                f=np.loadtxt(fname)
                ret=dict([(n,f[:,i]) for n,i in zip(fields,range(f.shape[1]))]);
            #at this point, we have 'ret', which is a dictionary.
            break # file was found, do not iterate over other directories
        except IOError:
            continue
    if len(ret)==0 and not quiet:
        print "readintfile: could not find file %s in given directories." % filename
    return ret

def writeintfile(qs, ints, errs, header, areas=None, filetype='intnorm'):
    """Save 1D scattering data to intnorm files.
    
    Inputs:
        qs: list of q values
        ints: list of intensity (scattering cross-section) values
        errs: list of error values
        header: header dictionary (only the key 'FSN' is used)
        areas [optional]: list of effective area values or None
        filetype: 'intnorm' to save 'intnorm%d.dat' files. 'intbinned' to
            write 'intbinned%d.dat' files. Case insensitive.
    """
    filename='%s%d.dat' % (filetype, header['FSN'])
    fid=open(filename,'wt');
    for i in range(len(qs)):
        if areas is None:
            fid.write('%e %e %e\n' % (qs[i],ints[i],errs[i]))
        else:
            fid.write('%e %e %e %e\n' % (qs[i],ints[i],errs[i],areas[i]))
    fid.close();
    
def write1dsasdict(data, filename):
    """Save 1D scattering data to file
    
    Inputs:
        data: 1D SAXS dictionary
        filename: filename
    """
    fid=open(filename,'wt');
    for i in range(len(data['q'])):
        fid.write('%e %e %e\n' % (data['q'][i],data['Intensity'][i],data['Error'][i]))
    fid.close();
    
def readintnorm(fsns, filetype='intnorm',dirs=[],logfiletype='intnorm',quiet=False,ext_types=['.npy','.mat','.txt','.dat']):
    """Read intnorm*.dat files along with their headers
    
    Inputs:
        fsns: one or more fsn-s.
        filetype: prefix of the filename
        logfiletype: prefix of the log filename
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed
        ext_types: extension types to try when loading file. Sequence and case
            is important!
        
    Outputs:
        A vector of dictionaries, in each dictionary the self-explanatory
            'q', 'Intensity' and 'Error' fields are present.
        A vector of parameters, read from the logfiles.
    
    Note:
        When loading only one fsn, the outputs will be still in lists, thus
            lists with one elements will be returned.
    """
    if not (isinstance(fsns,list) or isinstance(fsns,tuple)):
        fsns=[fsns];
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    data=[];
    param=[];
    for fsn in fsns:
        currdata={}
        currlog={}
        for d in dirs:
            filename='%s%s%s%d.dat' % (d,os.sep,filetype, fsn)
            tmp=readintfile(filename,quiet=quiet)
            if len(tmp)>0:
                currdata=tmp
                break # file was already found, do not try in another directory
        currlog=readlogfile(fsn,dirs,norm=logfiletype,quiet=quiet)
        if len(currdata)>0 and len(currlog)>0:
            data.append(currdata);
            param.append(currlog[0]);
    return data,param
def readbinned(fsn,dirs=[],quiet=False):
    """Read intbinned*.dat files along with their headers.
    
    This is a shortcut to readintnorm(fsn,'intbinned',dirs)
    """
    return readintnorm(fsn,'intbinned',dirs,quiet=quiet);
def readsummed(fsn,**kwargs):
    """Read summed*.dat files along with their headers.
    
    All arguments are forwarded to readintnorm().
    """
    return readintnorm(fsn,filetype='summed',**kwargs)
def readunited(fsn,**kwargs):
    """Read united*.dat files along with their headers.
    
    All arguments are forwarded to readintnorm().
    """
    return readintnorm(fsn,filetype='united',**kwargs)
    
def readlogfile(fsn,dirs=[],norm=True,quiet=False):
    """Read logfiles.
    
    Inputs:
        fsn: the file sequence number(s). It is possible to
            give a single value or a list
        dirs [optional]: a list of directories to try
        norm: if a normalized file is to be loaded (intnorm*.log). If
            False, intarb*.log will be loaded instead. Or, you can give a
            string. In that case, '%s%d.log' %(norm, <FSN>) will be loaded.
        quiet: True if no warning messages should be printed
            
    Output:
        a list of dictionaries corresponding to the header files. This
            is a list with one element if only one fsn was given. Thus the
            parameter dictionary will be params[0].
    """
    # this dictionary contains the floating point parameters. The key (first)
    # part of each item is the text before the value, up to (not included) the
    # colon. Ie. the key corresponding to line "FSN: 123" is 'FSN'. The value
    # (second) part of each item is the field (key) name in the resulting param
    # dictionary. If two float params are to be read from the same line (eg. the
    # line "Beam size X Y: 123.45, 135.78", )
    logfile_dict_float={'FSN':'FSN',
                        'Sample-to-detector distance (mm)':'Dist',
                        'Sample thickness (cm)':'Thickness',
                        'Sample transmission':'Transm',
                        'Sample position (mm)':'PosSample',
                        'Temperature':'Temperature',
                        'Measurement time (sec)':'MeasTime',
                        'Scattering on 2D detector (photons/sec)':'ScatteringFlux',
                        'Dark current subtracted (cps)':'dclevel',
                        'Dark current FSN':'FSNdc',
                        'Empty beam FSN':'FSNempty',
                        'Glassy carbon FSN':'FSNref1',
                        'Glassy carbon thickness (cm)':'Thicknessref1',
                        'Energy (eV)':'Energy',
                        'Calibrated energy (eV)':'EnergyCalibrated',
                        'Calibrated energy':'EnergyCalibrated',
                        'Beam x y for integration':('BeamPosX','BeamPosY'),
                        'Normalisation factor (to absolute units)':'NormFactor',
                        'Relative error of normalisation factor (percentage)':'NormFactorRelativeError',
                        'Beam size X Y (mm)':('BeamsizeX','BeamsizeY'),
                        'Pixel size of 2D detector (mm)':'PixelSize',
                        'Primary intensity at monitor (counts/sec)':'Monitor',
                        'Primary intensity calculated from GC (photons/sec/mm^2)':'PrimaryIntensity',
                        'Sample rotation around x axis':'RotXsample',
                        'Sample rotation around y axis':'RotYsample'
                        }
    #this dict. contains the string parameters
    logfile_dict_str={'Sample title':'Title',
                      'Sample name':'Title'}
    #this dict. contains the bool parameters
    logfile_dict_bool={'Injection between Empty beam and sample measurements?':'InjectionEB',
                       'Injection between Glassy carbon and sample measurements?':'InjectionGC'
                       }
    logfile_dict_list={'FSNs':'FSNs'}
    #some helper functions
    def getname(linestr):
        return string.strip(linestr[:string.find(linestr,':')]);
    def getvaluestr(linestr):
        return string.strip(linestr[(string.find(linestr,':')+1):])
    def getvalue(linestr):
        return float(getvaluestr(linestr))
    def getfirstvalue(linestr):
        valuepart=getvaluestr(linestr)
        return float(valuepart[:string.find(valuepart,' ')])
    def getsecondvalue(linestr):
        valuepart=getvaluestr(linestr)
        return float(valuepart[(string.find(valuepart,' ')+1):])
    def getvaluebool(linestr):
        valuepart=getvaluestr(linestr)
        if string.find(valuepart,'n')>=0:
            return False
        elif string.find(valuepart,'y')>0:
            return True
        else:
            return None
    #this is the beginning of readlogfile().
    if isinstance(norm,bool):
        if norm:
            norm='intnorm'
        else:
            norm='intarb'
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if np.isscalar(fsn):
        fsn=[fsn]
    params=[]; #initially empty
    for f in fsn:
        filefound=False
        for d in dirs:
            filebasename='%s%d.log' % (norm,f) #the name of the file
            filename='%s%s%s' %(d,os.sep,filebasename)
            try:
                param={};
                fid=open(filename,'r'); #try to open. If this fails, an exception is raised
                lines=fid.readlines(); # read all lines
                fid.close(); #close
                del fid;
                for line in lines:
                    name=getname(line);
                    for k in logfile_dict_float.keys():
                        if name==k:
                            if isinstance(logfile_dict_float[k],str):
                                param[logfile_dict_float[k]]=getvalue(line);
                            else: # type(logfile_dict_float[k]) is types.TupleType
                                param[logfile_dict_float[k][0]]=getfirstvalue(line);
                                param[logfile_dict_float[k][1]]=getsecondvalue(line);
                    for k in logfile_dict_str.keys():
                        if name==k:
                            param[logfile_dict_str[k]]=getvaluestr(line);
                    for k in logfile_dict_bool.keys():
                        if name==k:
                            param[logfile_dict_bool[k]]=getvaluebool(line);
                    for k in logfile_dict_list.keys():
                        if name==k:
                            spam=getvaluestr(line).split()
                            shrubbery=[]
                            for x in spam:
                                try:
                                    shrubbery.append(float(x))
                                except:
                                    shrubbery.append(x)
                            param[logfile_dict_list[k]]=shrubbery
                param['Title']=string.replace(param['Title'],' ','_');
                param['Title']=string.replace(param['Title'],'-','_');
                params.append(param);
                filefound=True
                del lines;
                break # file was already found, do not try in another directory
            except IOError:
                #print 'Cannot find file %s.' % filename
                pass
        if not filefound and not quiet:
            print 'Cannot find file %s in any of the given directories.' % filebasename
    return params;
            
    
def readwaxscor(fsns,dirs=[]):
    """Read corrected waxs file
    
    Inputs:
        fsns: a range of fsns or a single fsn.
        dirs [optional]: a list of directories to try
        
    Output:
        a list of scattering data dictionaries (see readintfile())
    """
    if not (isinstance(fsns,list) or isinstance(fsns,tuple)):
        fsns=[fsns]
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    waxsdata=[];
    for fsn in fsns:
        filefound=False
        for d in dirs:
            try:
                filename='%s%swaxs_%05d.cor' % (d,os.sep,fsn)
                tmp=np.loadtxt(filename)
                if tmp.shape[1]==3:
                    tmp1={'q':tmp[:,0],'Intensity':tmp[:,1],'Error':tmp[:,2]}
                waxsdata.append(tmp1)
                filefound=True
                break # file was found, do not try in further directories
            except IOError:
                pass
                #print '%s not found. Skipping it.' % filename
        if not filefound:
            print 'File waxs_%05d.cor was not found. Skipping.' % fsn
    return waxsdata

def readenergyfio(filename,files,fileend,dirs=[]):
    """Read abt_*.fio files.
    
    Inputs:
        filename: beginning of the file name, eg. 'abt_'
        files: a list or a single fsn number, eg. [1, 5, 12] or 3
        fileend: extension of a file, eg. '.fio'
        dirs [optional]: a list of directories to try
    
    Outputs: three lists:
        energies: the uncalibrated (=apparent) energies for each fsn.
        samples: the sample names for each fsn
        muds: the mu*d values for each fsn
    """
    if not (isinstance(files,list) or isinstance(files,tuple)):
        files=[files]
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    samples=[]
    energies=[]
    muds=[]
    for f in files:
        filefound=False
        for d in dirs:
            mud=[];
            energy=[];
            if isinstance(f,str):
                fname='%s%s%s%s%s' % (d,os.sep,filename,f,fileend)
            else:
                fname='%s%s%s%05d%s' % (d,os.sep,filename,f,fileend)
            try:
                fid=open(fname,'r')
                lines=fid.readlines()
                samples.append(lines[5].strip())
                for l in lines[41:]:
                    tmp=l.strip().split()
                    if len(tmp)==11:
                        try:
                            tmpe=float(tmp[0])
                            tmpmud=float(tmp[-1])
                            energy.append(tmpe)
                            mud.append(tmpmud)
                        except ValueError:
                            pass
                muds.append(mud)
                energies.append(energy)
                filefound=True
                break #file found, do not try further directories
            except IOError:
                pass
        if not filefound:
            print 'Cannot find file %s%05d%S.' % (filename,f,fileend)
    return (energies,samples,muds)

def readxanes(filebegin,files,fileend,energymeas,energycalib,dirs=[]):
    """Read energy scans from abt_*.fio files by readenergyfio() then
    put them on a correct energy scale.
    
    Inputs:
        filebegin: the beginning of the filename, like 'abt_'
        files: FSNs, like range(2,36)
        fileend: the end of the filename, like '.fio'
        energymeas: list of the measured energies
        energycalib: list of the true energies corresponding to the measured
            ones
        dirs [optional]: a list of directories to try
    
    Output:
        a list of mud dictionaries. Each dict will have the following items:
            'Energy', 'Mud', 'Title', 'scan'. The first three are
            self-describing. The last will be the FSN.
    """
    muds=[];
    if not (isinstance(files,list) or isinstance(files,tuple)):
        files=[files]

    for f in files:
        energy,sample,mud=readenergyfio(filebegin,f,fileend,dirs)
        if len(energy)>0:
            d={}
            d['Energy']=energycalibration(energymeas,energycalib,np.array(energy[0]))
            d['Mud']=np.array(mud[0])
            d['Title']=sample[0]
            d['scan']=f
            muds.append(d);
    return muds

def readabt(filename,dirs='.'):
    """Read abt_*.fio type files.
    
    Input:
        filename: the name of the file.
        dirs: directories to search for files in
        
    Output:
        A dictionary with the following fields:
            'title': the sample title
            'mode': 'Energy' or 'Motor'
            'columns': the description of the columns in 'data'
            'data': the data found in the file, in a matrix.
            'dataset': a structured array a la numpy, containing the same data
                as in 'data', but in another representation.
    """
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    ret={}
    for d in dirs:
        try:
            f=open(os.path.join(d,filename),'rt');
        except IOError:
            print 'Cannot open file %s' % filename
            continue
        # now the file is open
        lines=f.readlines()
        #prune comment lines starting with an exclamation mark (!).
        i=0
        while i<len(lines):
            lines[i]=lines[i].strip()
            if lines[i].startswith('!') or len(lines[i])==0:
                lines.pop(i)
                i-=1
            i+=1
        # find the parameter part
        ret['params']={}
        idx=lines.index('%p')+1
        while idx<len(lines) and (not lines[idx].startswith('%')):
            ls=lines[idx].split('=')
            if len(ls)==2:
                ret['params'][ls[0].strip()]=float(ls[1].strip())
            idx+=1
        # find the comment part
        idx=lines.index('%c')
        # first comment line is like: MOT12-Scan started at 21-Sep-2009 13:43:56, ended 13:47:53
        l=lines[idx+1]
        if l.startswith('MOT'):
            ret['mode']='Motor'
        elif l.startswith('ENERGY'):
            ret['mode']='Energy'
        else:
            ret['mode']='Motor'
            #print l
            #print 'Unknown scan type!'
            #return None
        # find the string containing the start time in dd-mmm-yyyy hh:mm:ss format
        stri=l[(l.index('started at ')+len('started at ')):l.index(', ended')]
        date,time1=stri.split(' ')
        ret['params']['day'],ret['params']['month'],ret['params']['year']=date.split('-')
        ret['params']['day']=int(ret['params']['day'])
        ret['params']['year']=int(ret['params']['year'])
        ret['params']['hourstart'],ret['params']['minutestart'],ret['params']['secondstart']=[int(x) for x in time1.split(':')]
        stri=l[l.index('ended')+len('ended '):]
        ret['params']['hourend'],ret['params']['minuteend'],ret['params']['secondend']=[int(x) for x in stri.split(':')]
        
        l=lines[idx+2]
        if l.startswith('Name:'):
            ret['name']=l.split(':')[1].split()[0]
        else:
            raise ValueError('File %s is invalid!' % filename)
        
        l=lines[idx+3]
        if l.startswith('Counter readings are'):
            ret['title']=''
            idx-=1
        else:
            ret['title']=l.strip()
        
        #idx+4 is "Counter readings are offset corrected..."
        l=lines[idx+5]
        if not l.startswith('%'):
            ret['offsets']={}
            lis=l.split()
            while len(lis)>0:
                ret['offsets'][lis.pop()]=float(lis.pop())
        idx=lines.index('%d')+1
        ret['columns']=[];
        while lines[idx].startswith('Col'):
            ret['columns'].append(lines[idx].split()[2][10:])
            idx+=1
        datalines=lines[idx:]
        for i in range(len(datalines)):
            datalines[i]=[float(x) for x in datalines[i].split()]
        ret['data']=np.array(datalines)
        ret['dataset']=np.array([tuple(a) for a in ret['data'].tolist()], dtype=zip(ret['columns'],[np.double]*len(ret['columns'])))
        return ret;
    return None

def readmask(filename,fieldname=None,dirs='.'):
    """Try to load a maskfile (matlab(R) matrix file)
    
    Inputs:
        filename: the input file name
        fieldname: field in the mat file. None to autodetect.
        dirs: list of directory names to try
        
    Outputs:
        the mask in a numpy array of type np.uint8
    """
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    f=None
    for d in dirs:
        try:
            f=scipy.io.loadmat(os.path.join(d,filename))
        except IOError:
            f=None
            continue
        else:
            break
    if f is None:
        raise IOError('Cannot find mask file in any of the given directories!')
    if fieldname is not None:
        return f[fieldname].astype(np.uint8)
    else:
        validkeys=[k for k in f.keys() if not (k.startswith('_') and k.endswith('_'))];
        if len(validkeys)<1:
            raise ValueError('mask file contains no masks!')
        if len(validkeys)>1:
            raise ValueError('mask file contains multiple masks!')
        return f[validkeys[0]].astype(np.uint8)
    
def writeparamfile(filename,param): #FIXME: this function is not complete yet
    """Write the param structure into a logfile. See writelogfile() for an explanation.
    
    Inputs:
        filename: name of the file.
        param: param structure (dictionary).
        
    Notes:
        exceptions pass through to the caller.
    """
    #information on how to store the param structure. Each sub-list corresponds to
    # a line in the param structure and should be of the form
    # [<beginning of the line>,<field name(s)>,[<formatter function>]]
    #
    # field name can be a string or a tuple of strings. Formatter function can
    # be a function accepting a single argument (the value of the field) or a
    # tuple of functions. If omitted, unicode() will be used.
    logfile_data=[('FSN','FSN'),
                  ('FSNs','FSNs',lambda l:' '.join([unicode(x) for x in l])),
                  ('Sample name','Title'),
                  ('Sample title','Title'),
                  ('Sample-to-detector distance (mm)','Dist'),
                  ('Sample thickness (cm)','Thickness'),
                  ('Sample transmission','Transm'),
                  ('Sample position (mm)','PosSample'),
                  ('Temperature','Temperature'),
                  ('Measurement time (sec)','MeasTime'),
                  ('Scattering on 2D detector (photons/sec)','ScatteringFlux'),
                  ('Dark current subtracted (cps)','dclevel'),
                  ('Dark current FSN','FSNdc'),
                  ('Empty beam FSN','FSNempty'),
                  ('Injection between Empty beam and sample measurements?','InjectionEB',lambda b:['n','y'][bool(b)]),
                  ('Glassy carbon FSN','FSNref1'),
                  ('Glassy carbon thickness (cm)','Thicknessref1'),
                  ('Injection between Glassy carbon and sample measurements?','InjectionGC',lambda b:['n','y'][bool(b)]),
                  ('Energy (eV)','Energy'),
                  ('Calibrated energy (eV)','EnergyCalibrated'),
                  ('Calibrated energy','EnergyCalibrated'),
                  ('Beam x y for integration',('BeamPosX','BeamPosY')),
                  ('Normalisation factor (to absolute units)','NormFactor'),
                  ('Relative error of normalisation factor (percentage)','NormFactorRelativeError'),
                  ('Beam size X Y (mm)',('BeamsizeX','BeamsizeY')),
                  ('Pixel size of 2D detector (mm)','PixelSize'),
                  ('Primary intensity at monitor (counts/sec)','Monitor'),
                  ('Primary intensity calculated from GC (photons/sec/mm^2)','PrimaryIntensity'),
                  ('Sample rotation around x axis','RotXsample'),
                  ('Sample rotation around y axis','RotYsample'),
                 ]
    allkeys=param.keys()
    f=open(filename,'wt')
    for ld in logfile_data:
        linebegin=ld[0]
        fieldnames=ld[1]
        if len(ld)<3:
            formatter=unicode
        else:
            formatter=ld[2]
        #some normalization
        if not isinstance(fieldnames,tuple):
            fieldnames=tuple([fieldnames,])
        if not isinstance(formatter,tuple):
            formatter=tuple([formatter,])
        if len(formatter)==1 and len(fieldnames)>1:
            formatter=tuple([formatter[0]]*len(fieldnames))
        if len(formatter)!=len(fieldnames):
            raise SyntaxError('logfile_data is badly written (line %s). This is a coding error.'%linebegin)
        #try to get the values
        try:
            vals=[f(param[k]) for f,k in zip(formatter,fieldnames)]
            f.write(linebegin+':\t'+' '.join(vals)+'\n')
        except:
            f.close()
            raise
        for fn in fieldnames:
            allkeys.remove(fn)
    for k in allkeys:
        f.write(k+':\t'+unicode(param[k])+'\n')
    f.close()
