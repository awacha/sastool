import os
import numpy as np
import datetime

def readasa(basename,dirs=[]):
    """Load SAXS/WAXS measurement files from ASA *.INF, *.P00 and *.E00 files.
    
    Input:
        basename: the basename (without extension) of the files. Can also be a
            list of strings
        dirs: list of directories (or just a single directory) to search files
            in. P00, INF and E00 should reside in the same directory.
    Output:
        An ASA dictionary (or a list of them) with the following fields:
            position: the counts for each pixel (numpy array), in cps
            energy: the energy spectrum (numpy array), in cps
            params: parameter dictionary. It has the following fields:
                Month: The month of the measurement
                Day: The day of the measurement
                Year: The year of the measurement
                Hour: The hour of the measurement
                Minute: The minute of the measurement
                Second: The second of the measurement
                Title: The title. If the user has written something to the
                    first line of the .INF file, it will be regarded as the
                    title. Otherwise the basename will be picked for this
                    field.
                Basename: The base name of the files (without the extension)
                Energywindow_Low: the lower value of the energy window
                Energywindow_High: the higher value of the energy window
                Stopcondition: stop condition in a string
                Realtime: real time in seconds
                Livetime: live time in seconds
                Datetime: date and time in a datetime.datetime struct.
            pixels: the pixel numbers.
            poserror: estimated error of the position (cps)
            energyerror: estimated error of the energy (cps)
    """
    if not (isinstance(dirs,list) or isinstance(dirs,tuple)):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if not (isinstance(basename,list) or isinstance(basename,tuple)):
        basenames=[basename]
        basename_scalar=True
    else:
        basenames=basename
        basename_scalar=False
    ret=[]
    for basename in basenames:
        for d in dirs:
            try:
                p00=np.loadtxt(os.path.join(d,'%s.P00' % basename))
            except IOError:
                try:
                    p00=np.loadtxt(os.path.join(d,'%s.p00' % basename))
                except:
                    p00=None
            if p00 is not None:
                p00=p00[1:] # cut the leading -1
            try:
                e00=np.loadtxt(os.path.join(d,'%s.E00' % basename))
            except IOError:
                try:
                    e00=np.loadtxt(os.path.join(d,'%s.e00' % basename))
                except:
                    e00=None
            if e00 is not None:
                e00=e00[1:] # cut the leading -1
            try:
                inffile=open(os.path.join(d,'%s.inf' % basename),'rt')
            except IOError:
                try:
                    inffile=open(os.path.join(d,'%s.Inf' % basename),'rt')
                except IOError:
                    try:
                        inffile=open(os.path.join(d,'%s.INF' % basename),'rt')
                    except:
                        inffile=None
                        params=None
            if (p00 is not None) and (e00 is not None) and (inffile is not None):
                break
            else:
                p00=None
                e00=None
                inffile=None
        if (p00 is None) or (e00 is None) or (inffile is None):
            print "Cannot find every file (*.P00, *.INF, *.E00) for sample %s in any directory" %basename
            continue
        if inffile is not None:
            params={}
            l1=inffile.readlines()
            l=[]
            for line in l1:
                if len(line.strip())>0:
                    l.append(line) # filter out empty lines
            def getdate(stri):
                try:
                    month=int(stri.split()[0].split('-')[0])
                    day=int(stri.split()[0].split('-')[1])
                    year=int(stri.split()[0].split('-')[2])
                    hour=int(stri.split()[1].split(':')[0])
                    minute=int(stri.split()[1].split(':')[1])
                    second=int(stri.split()[1].split(':')[2])
                except:
                    return None
                return {'Month':month,'Day':day,'Year':year,
                        'Hour':hour,'Minute':minute,'Second':second,
                        'Datetime':datetime.datetime(year,month,day,hour,minute,second)}
            #Three different cases can exist:
            #    1) Original, untouched INF file: first row is the date, second starts with Resolution
            #    2) Comments before the date
            #    3) Comments after the date
            resolutionlinepassed=False
            commentlines=[]
            for line in l:
                line=line.replace('\r','')
                if line.strip().startswith('Resolution'):
                    resolutionlinepassed=True
                elif line.strip().startswith('PSD1 Lower Limit'):
                    params['Energywindow_Low']=float(line.strip().split(':')[1].replace(',','.'))
                elif line.strip().startswith('PSD1 Upper Limit'):
                    params['Energywindow_High']=float(line.strip().split(':')[1].replace(',','.'))
                elif line.strip().startswith('Realtime'):
                    params['Realtime']=float(line.strip().split(':')[1].split()[0].replace(',','.').replace('\xa0',''))
                elif line.strip().startswith('Lifetime'):
                    params['Livetime']=float(line.strip().split(':')[1].split()[0].replace(',','.').replace('\xa0',''))
                elif line.strip().startswith('Lower Limit'):
                    params['Energywindow_Low']=float(line.strip().split(':')[1].replace(',','.'))
                elif line.strip().startswith('Upper Limit'):
                    params['Energywindow_High']=float(line.strip().split(':')[1].replace(',','.'))
                elif line.strip().startswith('Stop Condition'):
                    params['Stopcondition']=line.strip().split(':')[1].strip().replace(',','.')
                elif getdate(line) is not None:
                    params.update(getdate(line))
                else:
                    if not resolutionlinepassed:
                        commentlines.append(line.strip())
            params['comment']='\n'.join(commentlines)
            params['comment']=params['comment'].decode('cp1252')
            params['Title']=params['comment']
            params['basename']=basename.split(os.sep)[-1]
        ret.append({'position':p00/params['Livetime'],'energy':e00/params['Livetime'],
                'params':params,'pixels':np.arange(len(p00)),
                'poserror':np.sqrt(p00)/params['Livetime'],
                'energyerror':np.sqrt(e00)/params['Livetime'],
                'vectors':['pixels','position','poserror'],
                'x_is':'pixels','y_is':'position','dy_is':'poserror'})
    if basename_scalar:
        return ret[0]
    else:
        return ret
