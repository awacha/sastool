import gzip

from sastool.io.twodim import readcbf,readjusifaorg,readtif
from sastool.misc import findfileindirs


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

def read2dB1data(fsns,fileformat='org_%d',dirs=[],quiet=False):
    """Read 2D measurement files, along with their header data

    Inputs:
        fsns: the file sequence number or a list (or tuple or set or np.ndarray 
            of them).
        fileformat: format of the file name (without the extension!)
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed
        
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
            data=readcbf(dataname)
        elif dataname.upper().endswith('.DAT') or dataname.upper().endswith('.DAT.GZ'):
            data=readjusifaorg(dataname)
        else:
            data=readtif(data)
        datas.append(data)
        headers.append(header)
    return datas,headers
