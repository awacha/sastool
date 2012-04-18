import numpy as np
import re
import itertools
import datetime
import glob
import os
import xlwt
import numbers

import twodim
from ..misc import findfileindirs, normalize_listargument
from ..dataset import SASCurve
from classes import SASHeader, SASExposure

def readB1header(filename):
    """Reads B1 header data
    
    Inputs:
        filename: the file name.
        
    Output:
        A list of header dictionaries. An empty list if no headers were read.
        
    Examples:
        read header data from 'ORG000123.DAT':
        
        header=readB1header('ORG',123,'.DAT')
        
        or
        
        header=readB1header('ORG00123.DAT')

        or
        
        header=readB1header('ORG%05d.DAT',123)
    """
    return SASHeader.new_from_B1_org(filename)

def read2dB1data(fsns,fileformat='org_%05d',dirs=[]):
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
        data,header=read2dB1data(range(123,131),'ORG%05d')
        b) measurements with a Pilatus* detector:
        #in this case the org_*.header files should be present
        data,header=read2dB1data(range(123,131),'org_%05d')
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

def read2dintfile(fsn,fileformat='int2dnorm%d',logfileformat='intnorm%d.log',dirs=[]):
    fsns=normalize_listargument(fsn)
    def read_and_eat_exception(f):
        try:
            return SASExposure.new_from_B1_int2dnorm(f,fileformat,logfileformat,dirs)
        except IOError:
            print "Could not load files for FSN",f
            return None
    loaded=[read_and_eat_exception(f) for f in fsns]
    loaded=[l for l in loaded if l is not None]
    if isinstance(fsn,numbers.Number):
        return loaded[0]
    return loaded

def readparamfile(filename):
    """Read param files (*.log)
    
    Inputs:
        filename: the file name

    Output: the parameter dictionary
    """
    return SASHeader.new_from_B1_log(filename)

def readlogfile(fsns,paramformat='intnorm%d.log',dirs=[]):
    fsns=normalize_listargument(fsns)
    logfiles=[]
    for f in fsns:
        try:
            logfiles.append(readparamfile(findfileindirs(paramformat%f,dirs)))
        except IOError:
            pass
    return logfiles

def writeparamfile(filename,param):
    """Write the param structure into a logfile. See writelogfile() for an explanation.
    
    Inputs:
        filename: name of the file.
        param: param structure (dictionary).
        
    Notes:
        all exceptions pass through to the caller.
    """
    return SASHeader(param).write_B1_log(filename)
    
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
        data.header=param
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

def getsamplenamesxls(fsns,xlsname,dirs,whattolist=None,headerformat='org_%05d.header'):
    """ getsamplenames revisited, XLS output.
    
    Inputs:
        fsns: FSN sequence
        xlsname: XLS file name to output listing
        dirs: either a single directory (string) or a list of directories, a la readheader()
        whattolist: format specifier for listing. Should be a list of tuples. Each tuple
            corresponds to a column in the worksheet, in sequence. The first element of
            each tuple is the column title, eg. 'Distance' or 'Calibrated energy (eV)'.
            The second element is either the corresponding field in the header dictionary
            ('Dist' or 'EnergyCalibrated'), or a tuple of them, eg. ('FSN', 'Title', 'Energy').
            If the column-descriptor tuple does not have a third element, the string
            representation of each field (str(param[i][fieldname])) will be written
            in the corresponding cell. If a third element is present, it is treated as a 
            format string, and the values of the fields are substituted.
        headerformat: C-style format string of header file names (e.g. org_%05d.header)
        
    Outputs:
        an XLS workbook is saved.
    
    Notes:
        if whattolist is not specified exactly (ie. is None), then the output
            is similar to getsamplenames().
        module xlwt is needed in order for this function to work. If it cannot
            be imported, the other functions may work, only this function will
            raise a NotImplementedError.
    """
    def readheader_swallow_exception(fsn):
        try:
            return readB1header(findfileindirs(headerformat%fsn,dirs))
        except IOError:
            return None
    params=[readheader_swallow_exception(f) for f in fsns]
    params=[a for a in params if a is not None];

    if whattolist is None:
        whattolist=[('FSN','FSN'),('Time','MeasTime'),('Energy','Energy'),
                    ('Distance','Dist'),('Position','PosSample'),
                    ('Transmission','Transm'),('Temperature','Temperature'),
                    ('Title','Title'),('Date',('Day','Month','Year','Hour','Minutes'),'%02d.%02d.%04d %02d:%02d')]
    wb=xlwt.Workbook(encoding='utf8')
    ws=wb.add_sheet('Measurements')
    for i in range(len(whattolist)):
        ws.write(0,i,whattolist[i][0])
    for i in range(len(params)):
        # for each param structure create a line in the table
        for j in range(len(whattolist)):
            # for each parameter to be listed, create a column
            if np.isscalar(whattolist[j][1]):
                # if the parameter is a scalar, make it a list
                fields=tuple([whattolist[j][1]])
            else:
                fields=whattolist[j][1]
            if len(whattolist[j])==2:
                if len(fields)>=2:
                    strtowrite=''.join([str(params[i][f]) for f in fields])
                else:
                    strtowrite=params[i][fields[0]]
            elif len(whattolist[j])>=3:
                strtowrite=whattolist[j][2] % tuple([params[i][f] for f in fields])
            else:
                assert False
            ws.write(i+1,j,strtowrite)
    wb.save(xlsname)
    