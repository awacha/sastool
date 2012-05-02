'''twodim.py

This module contains the most read/write code for two-dimensional scattering
data. Reader functions should generally have the signature

def read<what>(filename)

while writers should be of the form

def write<what>(filename,data)
'''

import os
import numpy as np
import datetime
import scipy.misc
import scipy.io
import re
import dateutil.parser

from _io import cbfdecompress # pylint : disable=E0611

from sastool.misc import normalize_listargument

def readjusifaorg(filename):
    return np.loadtxt(filename,skiprows=133)

def readyellowsubmarine(nameformat,fsns=None,dirs='.'):
    """Read data measured at the SANS instrument (aka. Yellow Submarine)
    of the Budapest Neutron Centre.

    Inputs:
        nameformat: C-style format string for the file name (e.g. 'XE%04d.DAT')
        fsns: file sequence numbers to interpolate the format string with. Can
            also be None. In that case, nameformat is treated as the complete
            filename
        dirs: directories to load the file from.

    Outputs: datas,params
        datas: list of 2D ndarrays
        params: 
    """
    if fsns is None:
        filenames=[nameformat]
    else:
        filenames=[nameformat%f for f in fsns]
    dirs=normalize_listargument(dirs)
    datas=[]
    params=[]
    for fn in filenames:
        for d in dirs:
            try:
                f=open(os.path.join(d,fn),'r')
            except IOError:
                continue
            try:
                s=f.read()            
                f.close()
                par={}
                par['FSN']=int(s[2:6])
                par['Owner']=s[6:12].strip()
                par['Title']=s[12:0x18].strip()
                par['MeasTime']=long(s[0x18:0x1e])
                par['Monitor']=long(s[0x1e:0x26])
                par['Day']=int(s[0x26:0x28])
                par['Month']=int(s[0x29:0x2b])
                par['Year']=int(s[0x2c:0x30])
                par['Hour']=int(s[0x30:0x32])
                par['Minute']=int(s[0x33:0x35])
                par['Second']=int(s[0x36:0x38])
                par['PosSample']=int(s[0x60:0x65])
                par['PosBS']=int(s[0x5b:0x60])
                par['PosDetector']=int(s[0x56:0x5b])
                par['max']=long(s[0x38:0x3d])
                par['selector_speed']=long(s[0x3d:0x42])
                par['WaveLength']=long(s[0x42:0x44])
                par['Dist_Ech_det']=long(s[0x44:0x49])
                par['comments']=re.sub(r'\s+',' ',s[0x6d:0x100].strip())
                par['sum']=long(s[0x65:0x6d])
                par['BeamPosX']=float(s[0x49:0x4d])
                par['BeamPosY']=float(s[0x4d:0x51])
                par['AngleBase']=float(s[0x51:0x56])
                par['Date']=datetime.datetime(par['Year'],par['Month'],par['Day'],par['Hour'],par['Minute'],par['Second'])
                par['Energy']=12398.419/par['WaveLength']
                params.append(par)
                datas.append(np.fromstring(s[0x100:],'>u2').astype(np.double).reshape((64,64)))
                break
            except ValueError:
                print "File %s is invalid! Skipping."%fn
                continue
    if fsns is None:
        return datas[0],params[0]
    else:
        return datas,params

def readcbf(name):
    """Read a cbf (crystallographic binary format) file from a Dectris Pilatus
        detector.
    
    Inputs:
        name: filename
    
    Output:
        a numpy array of the scattering data
        
    Notes:
        currently only Little endian, "signed 32-bit integer" type and
        byte-offset compressed data are accepted.
    """
    def getvaluefromheader(header,caption,separator=':'):
        tmp=[x.split(separator)[1].strip() for x in header if x.startswith(caption)]
        if len(tmp)==0:
            raise ValueError ('Caption %s not present in CBF header!'%caption)
        else:
            return tmp[0]
    def cbfdecompress_old(data,dim1,dim2):
        index_input=0
        index_output=0
        value_current=0
        value_diff=0
        nbytes=len(data)
        output=np.zeros((dim1*dim2),np.double)
        while(index_input < nbytes):
            value_diff=ord(data[index_input])
            index_input+=1
            if value_diff !=0x80:
                if value_diff>=0x80:
                    value_diff=value_diff -0x100
            else: 
                if not ((ord(data[index_input])==0x00 ) and 
                    (ord(data[index_input+1])==0x80)):
                    value_diff=ord(data[index_input])+\
                                0x100*ord(data[index_input+1])
                    if value_diff >=0x8000:
                        value_diff=value_diff-0x10000
                    index_input+=2
                else:
                    index_input+=2
                    value_diff=ord(data[index_input])+\
                               0x100*ord(data[index_input+1])+\
                               0x10000*ord(data[index_input+2])+\
                               0x1000000*ord(data[index_input+3])
                    if value_diff>=0x80000000L:
                        value_diff=value_diff-0x100000000L
                    index_input+=4
            value_current+=value_diff
#            print index_output
            try:
                output[index_output]=value_current
            except IndexError:
                print "End of output array. Remaining input bytes:", len(data)-index_input
                print "remaining buffer:",data[index_input:]
                break
            index_output+=1
        if index_output != dim1*dim2:
            print "index_output is ",index_output-1
            print "dim1 is",dim1
            print "dim2 is",dim2
            print "dim1*dim2 is",dim1*dim2
            raise ValueError, "Binary data does not have enough points."
        return output.reshape((dim2,dim1))
    f=open(name,'rb')
    cbfbin=f.read()
    f.close()
    datastart=cbfbin.find('%c%c%c%c'%(12,26,4,213))+4
    header=[x.strip() for x in cbfbin[:datastart].split('\n')]
    if getvaluefromheader(header,'X-Binary-Element-Type')!='"signed 32-bit integer"':
        raise NotImplementedError('element type is not "signed 32-bit integer" in CBF, but %s.' % getvaluefromheader(header,'X-Binary-Element-Type'))
    if getvaluefromheader(header,'conversions','=')!='"x-CBF_BYTE_OFFSET"':
        raise NotImplementedError('compression is not "x-CBF_BYTE_OFFSET" in CBF!')
    dim1=long(getvaluefromheader(header,'X-Binary-Size-Fastest-Dimension'))
    dim2=long(getvaluefromheader(header,'X-Binary-Size-Second-Dimension'))
    nbytes=long(getvaluefromheader(header,'X-Binary-Size'))
    return cbfdecompress(cbfbin[datastart:datastart+nbytes],dim1,dim2)

def bdf_read(filename):
    """Read bdf file (Bessy Data Format)

    Input:
        filename: the name of the file

    Output:
        bdf: the BDF structure

    Adapted the bdf_read.m macro from Sylvio Haas.
    """
    bdf={}
    bdf['his']=[] #empty list for history
    bdf['C']={} # empty list for bdf file descriptions
    bdf['M']={} # empty list for motor positions
    bdf['CS']={} # empty list for scan parameters
    bdf['CT']={} # empty list for transmission data
    bdf['CG']={} # empty list for gain values
    mne_list=[]; mne_value=[]
    gain_list=[]; gain_value=[]
    s_list=[]; s_value=[]
    t_list=[]; t_value=[]
    
    fid=open(filename,'rb') #if fails, an exception is raised
    line=fid.readline()
    while len(line)>0:
        mat=line.split()
        if len(mat)==0:
            line=fid.readline()
            continue
        prefix=mat[0]
        sz=len(mat)
        if prefix=='#C':
            if sz==4:
                if mat[1]=='xdim':
                    bdf['xdim']=float(mat[3])
                elif mat[1]=='ydim':
                    bdf['ydim']=float(mat[3])
                elif mat[1]=='type':
                    bdf['type']=mat[3]
                elif mat[1]=='bdf':
                    bdf['bdf']=mat[3]
                elif mat[2]=='=':
                    bdf['C'][mat[1]]=mat[3]
            else:
                if mat[1]=='Sample':
                    bdf['C']['Sample']=[mat[3:]]
        if prefix[:4]=="#CML":
            mne_list.extend(mat[1:])
        if prefix[:4]=="#CMV":
            mne_value.extend(mat[1:])
        if prefix[:4]=="#CGL":
            gain_list.extend(mat[1:])
        if prefix[:4]=="#CGV":
            gain_value.extend(mat[1:])
        if prefix[:4]=="#CSL":
            s_list.extend(mat[1:])
        if prefix[:4]=="#CSV":
            s_value.extend(mat[1:])
        if prefix[:4]=="#CTL":
            t_list.extend(mat[1:])
        if prefix[:4]=="#CTV":
            t_value.extend(mat[1:])
        if prefix[:2]=="#H":
            szp=len(prefix)+1
            tline='%s' % line[szp:]
            bdf['his'].append(tline)

        if line[:5]=='#DATA':
            darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
            bdf['data']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy() # this weird transformation is needed to get the matrix in the same form as bdf_read.m gets it.
        if line[:6]=='#ERROR':
            darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
            bdf['error']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy()
        line=fid.readline()
    if len(mne_list)==len(mne_value):
        for j in range(len(mne_list)):
            bdf['M'][mne_list[j]]=mne_value[j]
    if len(gain_list)==len(gain_value):
        for j in range(len(gain_list)):
            bdf['CG'][gain_list[j]]=gain_value[j]
    if len(s_list)==len(s_value):
        for j in range(len(s_list)):
            bdf['CS'][s_list[j]]=s_value[j]
    if len(t_list)==len(t_value):
        for j in range(len(t_list)):
            bdf['CT'][t_list[j]]=t_value[j]
    fid.close()
    return bdf

def readtif(filename):
    """Read image files (TIFF, JPEG, PNG... supported by PIL).
    
    scipy.misc.imread() is used, which in turn depends on PIL.
    """
    return scipy.misc.imread(filename,True)

def readint2dnorm(filename):
    """Read corrected intensity and error matrices (Matlab mat or numpy npz
    format for Beamline B1 (HASYLAB/DORISIII)
    
    Outputs the intensity matrix and the error matrix
    
    File formats supported (ending of the filename):
        '.mat': Matlab MAT file, with (at least) two fields: Intensity and Error
        '.npz': Numpy zip file, with (at least) two fields: Intensity and Error
        other : the file is opened with np.loadtxt. The error matrix is tried to
            be loaded from the file <name>_error<ext> where the intensity was
            loaded from file <name><ext>. I.e. if 'somedir/matrix.dat' is given,
            the existence of 'somedir/matrix_error.dat' is checked. If not
            found, None is returned for the error matrix.
    
    The non-existence of the Intensity matrix results in an exception. If the
        Error matrix does not exist, None is returned for it.
    """
    # the core of read2dintfile
    if filename.upper().endswith('.MAT'): #Matlab
        m=scipy.io.loadmat(filename)
    elif filename.upper().endswith('.NPZ'): #Numpy
        m=np.load(filename)
    else: #loadtxt
        m={'Intensity':np.loadtxt(filename)}
        name,ext=os.path.splitext(filename)
        errorfilename=name+'_error'+ext
        if os.path.exists(errorfilename):
            m['Error']=np.loadtxt(errorfilename)
    try:
        Intensity=m['Intensity']
    except:
        raise
    try:
        Error=m['Error']
        return Intensity,Error
    except:
        return Intensity,None

def writeint2dnorm(filename,Intensity,Error=None):
    """Save the intensity and error matrices to a file
    
    Inputs:
        filename: the name of the file
        Intensity: the intensity matrix
        Error: the error matrix (can be None, if no error matrix is to be saved)
    """
    whattosave={'Intensity':Intensity}
    if Error is not None:
        whattosave['Error']=Error
    if filename.upper().endswith('.NPZ'):
        np.savez(filename, **whattosave)
    elif filename.upper().endswith('.MAT'):
        scipy.io.savemat(filename,whattosave)
    else: #text file
        np.savetxt(filename,Intensity)
        if Error is not None:
            name,ext=os.path.splitext(filename)
            np.savetxt(name+'_error'+ext,Error)

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
    
def _readedf_extractline(left, right):
    functions=[int, float, lambda l:float(l.split(None,1)[0]),
               lambda l:int(l.split(None,1)[0]),
               lambda l:dateutil.parser.parse(l), unicode]
    for f in functions:
        try:
            right=f(right)
            break;
        except ValueError:
            continue
    return right

def readehf(filename):
    f=open(filename,'r')
    edf={}
    if not f.readline().strip().startswith('{'):
        raise ValueError('Invalid file format.')
    for l in f:
        l=l.strip()
        if not l: continue
        if l.endswith('}'): break #last line of header
        try:
            left,right=l.split('=',1)
        except ValueError:
            raise ValueError('Invalid line: '+l)
        left=left.strip(); right=right.strip()
        if not right.endswith(';'):
            raise ValueError('Invalid line (does not end with a semicolon): '+l)
        right=right[:-1].strip()
        m=re.match('^(?P<left>.*)~(?P<continuation>\d+)$',left)
        if m is not None:
            edf[m.group('left')]=edf[m.group('left')]+right
        else:
            edf[left]=_readedf_extractline(left,right)
    f.close()
    edf['FileName']=filename
    return edf

def readedf(filename):
    edf=readehf(filename)
    f=open(filename,'rb')
    f.read(edf['EDF_HeaderSize'])  # skip header.
    if edf['DataType']=='FloatValue':
        dtype=np.float32
    else:
        raise NotImplementedError('Not supported data type: %s'%edf['DataType'])
    edf['data']=np.fromstring(f.read(edf['EDF_BinarySize']),dtype).reshape(edf['Dim_1'],edf['Dim_2'])
    return edf

def rebinmask(mask, binx, biny, enlarge=False):
    """Re-bin (shrink or enlarge) mask matrix.
    
    Inputs:
        mask: numpy array (dtype: np.uint8) of mask matrix. One is nonmasked,
            zero is masked.
        binx: binning along the 0th axis (integer)
        biny: binning along the 1st axis (integer)
        enlarge: direction of binning. If True, the matrix will be enlarged,
            otherwise shrinked (this is the default)
    
    Output:
        the binned mask matrix, of shape M/binx times N/biny or M*binx times
            N*biny(original mask is M times N pixels).
    """
    if not enlarge and ( (mask.shape[0] % binx) or (mask.shape[1] % biny)):
        raise ValueError('The number of pixels of the mask matrix should be divisible by the binning in each direction!')
    if enlarge:
        return mask.repeat(binx,axis=0).repeat(biny,axis=1)
    else:
        return mask[::binx,::biny]


