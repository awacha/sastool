'Input-output routines'
import scipy.io
import numpy as np
import os

from sastool.misc import energycalibration

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units


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
    
