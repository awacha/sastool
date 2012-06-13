'''
Procedures for reading/writing header metadata of exposures.
'''

import re
import dateutil.parser
import datetime
import numpy as np
import gzip

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
    #Planck's constant times speed of light: incorrect
    # constant in the old program on hasjusi1, which was
    # taken over by the measurement program, to keep
    # compatibility with that.
    hed={}
    jusifaHC = 12396.4
    if filename.upper().endswith('.GZ'):
        fid = gzip.GzipFile(filename, 'r')
    else:
        fid = open(filename, 'rt')
    lines = fid.readlines()
    fid.close()
    hed['FSN'] = int(lines[0].strip())
    hed['Hour'] = int(lines[17].strip())
    hed['Minutes'] = int(lines[18].strip())
    hed['Month'] = int(lines[19].strip())
    hed['Day'] = int(lines[20].strip())
    hed['Year'] = int(lines[21].strip()) + 2000
    hed['FSNref1'] = int(lines[23].strip())
    hed['FSNdc'] = int(lines[24].strip())
    hed['FSNsensitivity'] = int(lines[25].strip())
    hed['FSNempty'] = int(lines[26].strip())
    hed['FSNref2'] = int(lines[27].strip())
    hed['Monitor'] = float(lines[31].strip())
    hed['Anode'] = float(lines[32].strip())
    hed['MeasTime'] = float(lines[33].strip())
    hed['Temperature'] = float(lines[34].strip())
    hed['BeamPosX'] = float(lines[36].strip())
    hed['BeamPosY'] = float(lines[37].strip())
    hed['Transm'] = float(lines[41].strip())
    hed['Wavelength'] = float(lines[43].strip())
    hed['Energy'] = jusifaHC / hed['Wavelength']
    hed['Dist'] = float(lines[46].strip())
    hed['XPixel'] = 1 / float(lines[49].strip())
    hed['YPixel'] = 1 / float(lines[50].strip())
    hed['Title'] = lines[53].strip().replace(' ', '_').replace('-', '_')
    hed['MonitorDORIS'] = float(lines[56].strip())  # aka. DORIS counter
    hed['Owner'] = lines[57].strip()
    hed['RotXSample'] = float(lines[59].strip())
    hed['RotYSample'] = float(lines[60].strip())
    hed['PosSample'] = float(lines[61].strip())
    hed['DetPosX'] = float(lines[62].strip())
    hed['DetPosY'] = float(lines[63].strip())
    hed['MonitorPIEZO'] = float(lines[64].strip())  # aka. PIEZO counter
    hed['BeamsizeX'] = float(lines[66].strip())
    hed['BeamsizeY'] = float(lines[67].strip())
    hed['PosRef'] = float(lines[70].strip())
    hed['Monochromator1Rot'] = float(lines[77].strip())
    hed['Monochromator2Rot'] = float(lines[78].strip())
    hed['Heidenhain1'] = float(lines[79].strip())
    hed['Heidenhain2'] = float(lines[80].strip())
    hed['Current1'] = float(lines[81].strip())
    hed['Current2'] = float(lines[82].strip())
    hed['Detector'] = 'Unknown'
    hed['PixelSize'] = (hed['XPixel'] + hed['YPixel']) / 2.0
    
    hed['AnodeError'] = math.sqrt(hed['Anode'])
    hed['TransmError'] = 0
    hed['MonitorError'] = math.sqrt(hed['Monitor'])
    hed['MonitorPIEZOError'] = math.sqrt(hed['MonitorPIEZO'])
    hed['MonitorDORISError'] = math.sqrt(hed['MonitorDORIS'])
    hed['Date'] = datetime.datetime(hed['Year'], hed['Month'], hed['Day'], hed['Hour'], hed['Minutes'])
    hed['Origin'] = 'B1 original header'
    return hed


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

def readbhfv2(filename):
    header={}
    f=open(filename,'rt')
    for l in f:
        if not l.startswith('#'):
            continue
        l=l[1:].strip()
        section,keyvalue=l.split(None,1)
        if section not in header.keys():
            if section in ['HIS']:
                header[section]=[]
            else:
                header[section]={}
        if section in ['HIS']:
            header[section].append(keyvalue)
        else:
            key,value=keyvalue.split('=')
            value=value.strip()
            try:
                value=float(value)
            except ValueError:
                pass
            header[section][key.strip()]=value
    f.close()
    header['xdim']=header['C']['xdim']
    header['ydim']=header['C']['ydim']
    header['type']=header['C']['type']
    return header

def writebhfv2(filename, bdf):
    f=open(filename,'wt')
    f.write('#C xdim = %d\n'%bdf['xdim'])
    f.write('#C ydim = %d\n'%bdf['ydim'])
    f.write('#C type = %s\n'%bdf['type'])
    for k in [x for x in bdf.keys() if isinstance(bdf[x],dict)]:
        f.write('-------------------- %s field --------------------\n'%k)
        for l in bdf[k].keys():
            f.write("#%s %s = %s\n"%(k,l,bdf[k][l]))
    if 'HIS' in bdf.keys():
        f.write('-------------------- History --------------------\n')
        for h in bdf['HIS']:
            f.write("#HIS %s\n" %h)
    f.close()

def readbhf(filename,load_data=False):
    """Read header data from bdf/bhf file (Bessy Data Format v1)

    Input:
        filename: the name of the file
        load_data: if the matrices are to be loaded
    
    Output:
        bdf: the BDF header structure

    Adapted the bdf_read.m macro from Sylvio Haas.
    """
    bdf={}
    bdf['his']=[] #empty list for history
    bdf['C']={} # empty list for bdf file descriptions
    namelists={}
    valuelists={}
    with open(filename,'rb') as fid: #if fails, an exception is raised
        for line in fid:
            if not line.strip():
                continue  #empty line
            mat=line.split(None,1)
            prefix=mat[0]
            if prefix=='#C':
                left,right=mat[1].split('=',1)
                left=left.strip()
                right=right.strip()
                if left in ['xdim','ydim']:
                    bdf[left]=int(right)
                elif left in ['type','bdf']:
                    bdf[left]=right
                if left in ['Sendtime']:
                    bdf['C'][left]=float(right)
                elif left in ['xdim','ydim']:
                    bdf['C'][left]=int(right)
                else:
                    bdf['C'][left]=right
            elif prefix.startswith("#H"):
                bdf['his'].append(mat[1])
            elif prefix.startswith("#DATA"):
                if not load_data:
                    break
                darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
                bdf['data']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy() # this weird transformation is needed to get the matrix in the same form as bdf_read.m gets it.
            elif prefix.startswith('#ERROR'):
                if not load_data:
                    break
                darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
                bdf['error']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy()
            else:
                for prf in ['M','G','S','T']:
                    if prefix.startswith('#C%sL'%prf):
                        if prf not in namelists: namelists[prf]=[]
                        namelists[prf].extend(mat[1].split())
                    elif prefix.startswith('#C%sV'%prf):
                        if prf not in valuelists: valuelists[prf]=[]
                        valuelists[prf].extend([float(x) for x in mat[1].split()])
    for dictname,prfname in zip(['M','CG','CS','CT'],['M','G','S','T']):
        bdf[dictname]=dict(zip(namelists[prf],valuelists[prf]))
    return bdf

def readPAXE(filename,load_data=False):
    f=open(filename,'r')
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
    par['Dist']=long(s[0x44:0x49])
    par['comments']=re.sub(r'\s+',' ',s[0x6d:0x100].strip())
    par['sum']=long(s[0x65:0x6d])
    par['BeamPosX']=float(s[0x49:0x4d])
    par['BeamPosY']=float(s[0x4d:0x51])
    par['AngleBase']=float(s[0x51:0x56])
    par['Date']=datetime.datetime(par['Year'],par['Month'],par['Day'],par['Hour'],par['Minute'],par['Second'])
    try:
        par['Energy']=12398.419/par['WaveLength']
    except ZeroDivisionError:
        par['Energy']=np.nan
    par['Detector']='XE'
    par['PixelSize']=1
    if load_data:
        if filename.endswith('32'):
            return par,np.fromstring(s[0x100:],'<i4').astype(np.double).reshape((64,64))
        else:
            return par,np.fromstring(s[0x100:],'>u2').astype(np.double).reshape((64,64))
    else:
        return par

