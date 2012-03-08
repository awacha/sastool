import numpy as np
import dateutil.parser
from ..misc import normalize_listargument,findfileindirs
import warnings

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
    basenames=normalize_listargument(basename)
    ret=[]
    for basename in basenames:
        p00=None
        for ext in ['P00','p00']:
            try:
                p00=np.loadtxt(findfileindirs(basename+'.'+ext,dirs))
                break
            except IOError:
                pass
        if p00 is not None:
            p00=p00[1:]
        e00=None
        for ext in ['E00','e00']:
            try:
                e00=np.loadtxt(findfileindirs(basename+'.'+ext,dirs))
                break
            except IOError:
                pass
        if e00 is not None:
            e00=e00[1:]
        inffile=None
        for ext in ['Inf','inf','INF']:
            try:
                inffile=open(findfileindirs(basename+'.'+ext,dirs),'rt')
                break
            except IOError:
                pass
        if (p00 is None) or (e00 is None) or (inffile is None):
            warnings.warn("Cannot find every file (*.P00, *.INF, *.E00) for sample %s." %basename)
            continue
        if inffile is not None:
            params={}
            mode=''
            params['comments']=[]
            floatright=lambda s:float(s.split(':')[1].replace(',','.'))
            for line in inffile:
                line=line.replace('\xa0','').strip()
                if not line:
                    mode=''
                    continue # skip empty lines
                try:
                    params['Date']=dateutil.parser.parse(line)
                except ValueError:
                    pass
                else:
                    continue
                if line.endswith(':'):
                    mode=line
                elif mode.startswith('Resol'):
                    pass
                elif mode.startswith('Energy Window'):
                    if line.startswith('PSD1 Lower Limit') or line.startswith('Lower Limit'):
                        params['Energywindow_Low']=floatright(line)
                    elif line.startswith('PSD1 Upper Limit') or line.startswith('Upper Limit'):
                        params['Energywindow_High']=floatright(line)
                    else:
                        pass
                elif mode.startswith('Comment'):
                    params['comments'].append(line)
                elif line.startswith('Realtime'):
                    params['Realtime']=float(line.split(':')[1].split()[0].replace(',','.'))
                elif line.startswith('Lifetime'):
                    params['Livetime']=float(line.split(':')[1].split()[0].replace(',','.'))
                elif line.startswith('PSD1 Energy Counts') or line.startswith('Energy Counts'):
                    params['EnergyCounts']=floatright(line)
                elif line.startswith('PSD1 Position Counts') or line.startswith('Position Counts'):
                    params['PositionCounts']=floatright(line)
                elif line.startswith('Stop Condition'):
                    params['Stopcondition']=line.split(':')[1].strip()
            params['basename']=basename
            if params['comments']:
                params['Title']=params['comments'].split()[0]
            else:
                params['Title']=params['basename']
        ret.append({'position':p00,'energy':e00,
                    'params':params,'pixels':np.arange(len(p00)),
                    'poserror':np.sqrt(p00),
                    'energyerror':np.sqrt(e00),
                    })
    return ret