import gzip
import numpy as np
import re
import itertools
import datetime
import glob
import os

import twodim
from ..misc import findfileindirs, normalize_listargument
from ..dataset import SASCurve

#information on how to store the param structure. Each sub-list corresponds to
# a line in the param structure and should be of the form
# [<linestart>,<field name(s)>,<formatter function>,<reader function>]
#
# Required are the first and second.
#
# linestart: the beginning of the line in the file, up to the colon.
#
# field name(s): field name can be a string or a tuple of strings.
#
# formatter function: can be (1) a function accepting a single argument (the 
#     value of the field) or (2) a tuple of functions or (3) None. In the latter
#     case and when omitted, unicode() will be used.
# 
# reader function: can be (1) a function accepting a string and returning as
#     much values as the number of field names is. Or if omitted, unicode() will
#     be used.

logfile_data = [('FSN', 'FSN', None, int),
                  ('FSNs', 'FSNs', lambda l:' '.join([unicode(x) for x in l]),
                   lambda s:[float(x) for x in s.replace(',', ' ').replace(';', ' ').split()]),
                  ('Sample name', 'Title'),
                  ('Sample title', 'Title'),
                  ('Sample-to-detector distance (mm)', 'Dist', None, float),
                  ('Sample thickness (cm)', 'Thickness', None, float),
                  ('Sample transmission', 'Transm', None, float),
                  ('Sample position (mm)', 'PosSample', None, float),
                  ('Temperature', 'Temperature', None, float),
                  ('Measurement time (sec)', 'MeasTime', None, float),
                  ('Scattering on 2D detector (photons/sec)', 'ScatteringFlux',
                   None, float),
                  ('Dark current subtracted (cps)', 'dclevel', None, float),
                  ('Dark current FSN', 'FSNdc', None, int),
                  ('Empty beam FSN', 'FSNempty', None, int),
                  ('Injection between Empty beam and sample measurements?',
                   'InjectionEB', lambda b:['n', 'y'][bool(b)],
                   lambda s:s.upper().startswith('Y')),
                  ('Glassy carbon FSN', 'FSNref1', None, int),
                  ('Glassy carbon thickness (cm)', 'Thicknessref1', None, float),
                  ('Injection between Glassy carbon and sample measurements?',
                   'InjectionGC', lambda b:['n', 'y'][bool(b)],
                   lambda s:s.upper().startswith('Y')),
                  ('Energy (eV)', 'Energy', None, float),
                  ('Calibrated energy (eV)', 'EnergyCalibrated', None, float),
                  ('Calibrated energy', 'EnergyCalibrated', None, float),
                  ('Beam x y for integration', ('BeamPosX', 'BeamPosY'), None,
                   lambda s: tuple([float(x) for x in s.replace(',',' ').replace(';',' ').split()])),
                  ('Normalisation factor (to absolute units)', 'NormFactor',
                   None, float),
                  ('Relative error of normalisation factor (percentage)',
                   'NormFactorRelativeError', None, float),
                  ('Beam size X Y (mm)', ('BeamsizeX', 'BeamsizeY'), None,
                   lambda s: tuple([float(x) for x in s.replace(',',' ').replace(';',' ').split()])),
                  ('Pixel size of 2D detector (mm)', 'PixelSize', None, float),
                  ('Primary intensity at monitor (counts/sec)', 'Monitor', None,
                   float),
                  ('Primary intensity calculated from GC (photons/sec/mm^2)',
                   'PrimaryIntensity', None, float),
                  ('Sample rotation around x axis', 'RotXsample', None, float),
                  ('Sample rotation around y axis', 'RotYsample', None, float),
                 ]

def readB1header(filename):
    """Reads B1 header data
    
    Inputs:
        filename: the file name.
        
    Output:
        A list of header dictionaries. An empty list if no headers were read.
        
    Examples:
        read header data from 'ORG000123.DAT':
        
        header=readheader('ORG',123,'.DAT')
        
        or
        
        header=readheader('ORG00123.DAT')

        or
        
        header=readheader('ORG%05d.DAT',123)
    """
    #Planck's constant times speed of light: incorrect
    # constant in the old program on hasjusi1, which was
    # taken over by the measurement program, to keep
    # compatibility with that.
    jusifaHC=12396.4
    if filename.upper().endswith('.GZ'):
        fid=gzip.GzipFile(filename,'rt')
    else:
        fid=open(filename,'rt')
    header={}
    lines=fid.readlines()
    fid.close()
    header['FSN']=int(lines[0].strip())
    header['Hour']=int(lines[17].strip())
    header['Minutes']=int(lines[18].strip())
    header['Month']=int(lines[19].strip())
    header['Day']=int(lines[20].strip())
    header['Year']=int(lines[21].strip())+2000
    header['FSNref1']=int(lines[23].strip())
    header['FSNdc']=int(lines[24].strip())
    header['FSNsensitivity']=int(lines[25].strip())
    header['FSNempty']=int(lines[26].strip())
    header['FSNref2']=int(lines[27].strip())
    header['Monitor']=float(lines[31].strip())
    header['Anode']=float(lines[32].strip())
    header['MeasTime']=float(lines[33].strip())
    header['Temperature']=float(lines[34].strip())
    header['Transm']=float(lines[41].strip())
    header['Energy']=jusifaHC/float(lines[43].strip())
    header['Dist']=float(lines[46].strip())
    header['XPixel']=1/float(lines[49].strip())
    header['YPixel']=1/float(lines[50].strip())
    header['Title']=lines[53].strip().replace(' ','_').replace('-','_')
    header['MonitorDORIS']=float(lines[56].strip())
    header['Owner']=lines[57].strip()
    header['Rot1']=float(lines[59].strip())
    header['Rot2']=float(lines[60].strip())
    header['PosSample']=float(lines[61].strip())
    header['DetPosX']=float(lines[62].strip())
    header['DetPosY']=float(lines[63].strip())
    header['MonitorPIEZO']=float(lines[64].strip())
    header['BeamsizeX']=float(lines[66].strip())
    header['BeamsizeY']=float(lines[67].strip())
    header['PosRef']=float(lines[70].strip())
    header['Monochromator1Rot']=float(lines[77].strip())
    header['Monochromator2Rot']=float(lines[78].strip())
    header['Heidenhain1']=float(lines[79].strip())
    header['Heidenhain2']=float(lines[80].strip())
    header['Current1']=float(lines[81].strip())
    header['Current2']=float(lines[82].strip())
    header['Detector']='Unknown'
    header['PixelSize']=(header['XPixel']+header['YPixel'])/2.0
    return header

def read2dB1data(fsns,fileformat='org_%d',dirs=[]):
    """Read 2D measurement files, along with their header data

    Inputs:
        fsns: the file sequence number or a list (or tuple or set or np.ndarray 
            of them).
        fileformat: format of the file name (without the extension!)
        dirs [optional]: a list of directories to try
        
    Outputs: datas, headers
        datas: A list of 2d scattering data matrices (numpy arrays)
        headers: A list of header data (Python dicts)
        
    Examples:
        Read FSN 123-130:
        a) measurements with the Gabriel detector:
        data,header=read2dB1data('ORG',range(123,131),'.DAT')
        b) measurements with a Pilatus* detector:
        #in this case the org_*.header files should be present in the same folder
        data,header=read2dB1data('org_',range(123,131),'.tif')
    """
    fsns=normalize_listargument(fsns)
    datas=[]
    headers=[]
    for f in fsns:
        #try to load header file
        headername=''
        for extn in ['.header','.DAT','.dat','.DAT.gz','.dat.gz']:
            try:
                headername=findfileindirs(fileformat%f+extn,dirs);
            except IOError:
                continue
        if not headername:
            continue #skip this file
        dataname=''
        for extn in ['.cbf','.tif','.tiff','.DAT','.DAT.gz','.dat','.dat.gz']:
            try:
                dataname=findfileindirs(fileformat%f+extn,dirs)
            except IOError:
                continue
        if not dataname:
            continue #skip this file
        header=readB1header(headername)
        if dataname.endswith('.cbf'):
            data=twodim.readcbf(dataname)
        elif dataname.upper().endswith('.DAT') or dataname.upper().endswith('.DAT.GZ'):
            data=twodim.readjusifaorg(dataname).reshape(256,256)
        else:
            data=twodim.readtif(dataname)
        datas.append(data)
        headers.append(header)
    return datas,headers

def read2dintfile(fsns,fileformat='int2dnorm%d',dirs=[]):
    fsns=normalize_listargument(fsns)
    ints=[]
    errs=[]
    for f in fsns:
        dataname=None
        for extn in ['.npy','.mat']:
            try:
                dataname=findfileindirs(fileformat%f+extn,dirs)
            except IOError:
                continue
        if not dataname:
            continue
        i,e=twodim.readint2dnorm(dataname)
        ints.append(i)
        errs.append(e)

def readparamfile(filename):
    """Read param files (*.log)
    
    Inputs:
        filename: the file name
            
    Output: the parameter dictionary
    """
    fid=open(filename,'r'); #try to open. If this fails, an exception is raised
    param={}
    for l in fid:
        try:
            ld=[ld for ld in logfile_data if l.split(':')[0].strip()==ld[0]][0]
        except IndexError:
            #line is not recognized.
            continue
        if len(ld)<4:
            reader=unicode
        else:
            reader=ld[3]
        vals=reader(l.split(':')[1].strip())
        if isinstance(ld[1],tuple):
            #more than one field names. The reader function should return a 
            # tuple here, a value for each field.
            if len(vals)!=len(ld[1]):
                raise ValueError('Cannot read %d values from line %s in file!'%(len(ld[1]),l))
            param.update(dict(zip(ld[1],vals)))
        else:
            param[ld[1]]=vals
    fid.close()
    return param;
    
        
def writeparamfile(filename,param):
    """Write the param structure into a logfile. See writelogfile() for an explanation.
    
    Inputs:
        filename: name of the file.
        param: param structure (dictionary).
        
    Notes:
        exceptions pass through to the caller.
    """
    allkeys=param.keys()
    f=open(filename,'wt')
    for ld in logfile_data: #process each line
        linebegin=ld[0]
        fieldnames=ld[1]
        #set the default formatter if it is not given
        if len(ld)<3:
            formatter=unicode
        elif ld[2] is None:
            formatter=unicode
        else:
            formatter=ld[2]
        #this will contain the formatted values.
        formatted=''
        if isinstance(fieldnames,basestring):
            #scalar field name, just one field. Formatter should be a callable.
            if fieldnames not in allkeys:
                #this field has already been processed
                continue
            try:
                formatted=formatter(param[fieldnames])
            except KeyError:
                #field not found in param structure
                continue
        elif isinstance(fieldnames,tuple):
            #more than one field names in a tuple. In this case, formatter can
            # be a tuple of callables...
            if all([(fn not in allkeys) for fn in fieldnames]):
                #if all the fields have been processed:
                continue
            if isinstance(formatter,tuple) and len(formatter)==len(fieldnames):
                formatted=' '.join([ft(param[fn]) for ft,fn in zip(formatter,fieldnames)])
            #...or a single callable...
            elif not isinstance(formatter,tuple):
                formatted=' '.join([formatter(param[fn]) for fn in fieldnames])
            #...otherwise raise an exception.
            else:
                raise SyntaxError('Programming error: formatter should be a scalar or a tuple\
 of the same length as the field names in logfile_data.')
        else: #fieldnames is neither a string, nor a tuple.
            raise SyntaxError('Invalid syntax (programming error) in logfile_data in writeparamfile().')
        #try to get the values
        f.write(linebegin+':\t'+formatted+'\n')
        if isinstance(fieldnames,tuple):
            for fn in fieldnames: #remove the params treated.
                if fn in allkeys:
                    allkeys.remove(fn)
        else:
            if fieldnames in allkeys:
                allkeys.remove(fieldnames)
    #write untreated params
    for k in allkeys:
        f.write(k+':\t'+unicode(param[k])+'\n')
    f.close()

def read1d(fsns,fileformat='intnorm%d.dat',paramformat='intnorm%d.log',dirs=[]):
    fsns=normalize_listargument(fsns)
    datas=[]
    params=[]
    for f in fsns:
        try:
            filename=findfileindirs(fileformat%f,dirs)
            paramname=findfileindirs(paramformat%f,dirs)
        except IOError:
            continue
        data=SASCurve.new_from_file(filename)
        param=readparamfile(paramname)
        datas.append(data)
        params.append(param)
    return datas,params

def readbinned(fsns,*args,**kwargs):
    return read1d(fsns,*args,fileformat='intbinned%d.dat',**kwargs)

def readsummed(fsns,*args,**kwargs):
    return read1d(fsns,*args,fileformat='summed%d.dat',paramformat='summed%d.log',**kwargs)

def readunited(fsns,*args,**kwargs):
    return read1d(fsns,*args,fileformat='united%d.dat',paramformat='united%d.log',**kwargs)

def readwaxscor(fsns,*args,**kwargs):
    return read1d(fsns,*args,fileformat='waxs%d.cor',**kwargs)

def readabt(filename,dirs='.'):
    """Read abt_*.fio type files.
    
    Input:
        filename: the name of the file.
        dirs: directories to search for files in
        
    Output:
        A dictionary. The fields are self-explanatory.
    """
    #resolve filename
    filename=findfileindirs(filename,dirs)
    f=open(filename,'rt')
    abt={'offsetcorrected':False,'params':{},'columns':[],'data':[],'title':'<no_title>',
         'offsets':{},'filename':filename};
    readingmode=''
    for l in f:
        l=l.strip()
        if l.startswith('!') or len(l)==0:
            continue
        elif l.startswith('%c'):
            readingmode='comments';
        elif l.startswith('%p'):
            readingmode='params';
        elif l.startswith('%d'):
            readingmode='data';
        elif readingmode=='comments':
            m=re.match(r'(?P<scantype>\w+)-Scan started at (?P<startdate>\d+-\w+-\d+) (?P<starttime>\d+:\d+:\d+), ended (?P<endtime>\d+:\d+:\d+)',l)
            if m:
                abt.update(m.groupdict());
                continue
            else:
                m=re.match(r'Name: (?P<name>\w+)',l)
                if m:
                    abt.update(m.groupdict());
                    m1=re.search(r'from (?P<from>\d+(?:.\d+)?)',l)
                    if m1: abt.update(m1.groupdict())
                    m1=re.search(r'to (?P<to>\d+(?:.\d+)?)',l)
                    if m1: abt.update(m1.groupdict())
                    m1=re.search(r'by (?P<by>\d+(?:.\d+)?)',l)
                    if m1: abt.update(m1.groupdict())
                    m1=re.search(r'sampling (?P<sampling>\d+(?:.\d+)?)',l)
                    if m1: abt.update(m1.groupdict())
                    continue
            if l.find('Counter readings are offset corrected')>=0:
                abt['offsetcorrected']=True
                readingmode='offsets'
                continue
            #if we reach here in 'comments' mode, this is the title line
            abt['title']=l
            continue
        elif readingmode=='offsets':
            m=re.findall(r'(\w+)\s(\d+(?:.\d+)?)',l)
            if m:
                abt['offsets'].update(dict(m))
                for k in abt['offsets']:
                    abt['offsets'][k]=float(abt['offsets'][k])
        elif readingmode=='params':
            abt['params'][l.split('=')[0].strip()]=float(l.split('=')[1].strip())
        elif readingmode=='data':
            if l.startswith('Col'):
                abt['columns'].append(l.split()[2])
            else:
                abt['data'].append([float(x) for x in l.split()])
    f.close()
    #some post-processing
    #remove common prefix from column names
    maxcolnamelen=max(len(c) for c in abt['columns'])
    for l in range(1,maxcolnamelen):
        if len(set([c[:l] for c in abt['columns']]))>1:
            break
    abt['columns']=[c[l-1:] for c in abt['columns']]
    #represent data as a structured array
    dt=np.dtype(zip(abt['columns'],itertools.repeat(np.double)))
    abt['data']=np.array(abt['data'],dtype=np.double).view(dt)
    #dates and times in datetime formats
    monthnames=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m,i in zip(monthnames,itertools.count(1)):
        abt['startdate']=abt['startdate'].replace(m,str(i))
    abt['startdate']=datetime.date(*reversed([int(x) for x in abt['startdate'].split('-')]))
    abt['starttime']=datetime.time(*[int(x) for x in abt['starttime'].split(':')])
    abt['endtime']=datetime.time(*[int(x) for x in abt['endtime'].split(':')])
    abt['start']=datetime.datetime.combine(abt['startdate'],abt['starttime'])
    if abt['endtime']<=abt['starttime']:
        abt['end']=datetime.datetime.combine(abt['startdate']+datetime.timedelta(1),abt['endtime'])
    else:
        abt['end']=datetime.datetime.combine(abt['startdate'],abt['endtime'])
    del abt['starttime'];    del abt['startdate'];    del abt['endtime']
    #convert some fields to float
    for k in ['from','to','by','sampling']:
        if k in abt:
            abt[k]=float(abt[k])
    #change space and dash in title to underscore
    abt['title']=abt['title'].replace('-','_').replace(' ','_')
    return abt

def listabtfiles(directory='.',fileformat='abt*.fio'):
    lis=glob.glob(os.path.join(directory,fileformat))
    for filename in sorted(lis):
        try:
            abt=readabt(filename,[''])
        except IOError:
            pass
        print abt['name'], abt['scantype'], abt['title'], abt['start'].isoformat(), abt['end'].isoformat()