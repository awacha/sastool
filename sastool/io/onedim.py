'''
Low-level I/O for 1D data
'''

import datetime
import numpy as np
import re
import itertools

from .. import misc

def readspec(filename):
    f = open(filename, 'rt')
    sf = {}
    sf['originalfilename'] = filename
    for l in f:
        if l.startswith('#F'):
            sf['filename'] = l[2:].strip()
        elif l.startswith('#E'):
            sf['epoch'] = long(l[2:].strip())
            sf['datetime'] = datetime.datetime.fromtimestamp(sf['epoch'])
        elif l.startswith('#D'):
            if 'scans' in sf.keys():
                sf['scans'][-1]['datestring'] = l[2:].strip()
            sf['datestring'] = l[2:].strip()
        elif l.startswith('#C'):
            if 'scans' in sf.keys():
                sf['scans'][-1]['comment'] = l[2:].strip()
            else:
                sf['comment'] = l[2:].strip()
        elif l.startswith('#O'):
            l = l.split(None, 1)[1]
            if 'motors' not in sf.keys():
                sf['motors'] = []
            sf['motors'].extend([x.strip() for x in l.split('  ')])
        elif l.startswith('#S'):
            if 'scans' not in sf.keys():
                sf['scans'] = []
            sf['scans'].append({})
            sf['scans'][-1]['number'] = long(l[2:].split()[0])
            sf['scans'][-1]['command'] = l[2:].split(None, 1)[1].strip()
            sf['scans'][-1]['data'] = []
        elif l.startswith('#T'):
            sf['scans'][-1]['countingtime'] = float(l[2:].split()[0])
            sf['scans'][-1]['scantimeunits'] = l[2:].split()[1].strip()
        elif l.startswith('#M'):
            sf['scans'][-1]['countingcounts'] = float(l[2:].split()[0])
        elif l.startswith('#G'):
            if 'G' not in sf['scans'][-1].keys():
                sf['scans'][-1]['G'] = []
            sf['scans'][-1]['G'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#P'):
            if 'positions' not in sf['scans'][-1].keys():
                sf['scans'][-1]['positions'] = []
            sf['scans'][-1]['positions'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#Q'):
            pass
        elif l.startswith('#N'):
            n = tuple(float(x) for x in l[2:].strip().split())
            if len(n) == 1:
                sf['scans'][-1]['N'] = n[0]
            else:
                sf['scans'][-1]['N'] = n
        elif l.startswith('#L'):
            sf['scans'][-1]['Columns'] = [x.strip() for x in l[3:].split('  ')]
        elif len(l.strip()) == 0:
            pass
        elif l.startswith('#'):
            pass
        else:
            sf['scans'][-1]['data'].append([float(x) for x in l.split()])
    if 'scans' not in sf:
        sf['scans'] = []
    for s in sf['scans']:
        if 'data' in s.keys():
            s1 = [tuple(d) for d in s['data']]
            s['data'] = np.array(s1, dtype=zip(s['Columns'], [np.double] * len(s['Columns'])))
        s['motors'] = sf['motors']
        if 'comment' not in s:
            s['comment'] = sf['comment']
        if 'positions' not in s:
            s['positions'] = [None ] * len(sf['motors'])
    return sf

def readabt(filename, dirs='.'):
    """Read abt_*.fio type files from beamline B1, HASYLAB.
    
    Input:
        filename: the name of the file.
        dirs: directories to search for files in
        
    Output:
        A dictionary. The fields are self-explanatory.
    """
    # resolve filename
    filename = misc.findfileindirs(filename, dirs)
    f = open(filename, 'rt')
    abt = {'offsetcorrected':False, 'params':{}, 'columns':[], 'data':[], 'title':'<no_title>',
         'offsets':{}, 'filename':filename};
    readingmode = ''
    for l in f:
        l = l.strip()
        if l.startswith('!') or len(l) == 0:
            continue
        elif l.startswith('%c'):
            readingmode = 'comments';
        elif l.startswith('%p'):
            readingmode = 'params';
        elif l.startswith('%d'):
            readingmode = 'data';
        elif readingmode == 'comments':
            m = re.match(r'(?P<scantype>\w+)-Scan started at (?P<startdate>\d+-\w+-\d+) (?P<starttime>\d+:\d+:\d+), ended (?P<endtime>\d+:\d+:\d+)', l)
            if m:
                abt.update(m.groupdict());
                continue
            else:
                m = re.match(r'Name: (?P<name>\w+)', l)
                if m:
                    abt.update(m.groupdict());
                    m1 = re.search(r'from (?P<from>\d+(?:.\d+)?)', l)
                    if m1: abt.update(m1.groupdict())
                    m1 = re.search(r'to (?P<to>\d+(?:.\d+)?)', l)
                    if m1: abt.update(m1.groupdict())
                    m1 = re.search(r'by (?P<by>\d+(?:.\d+)?)', l)
                    if m1: abt.update(m1.groupdict())
                    m1 = re.search(r'sampling (?P<sampling>\d+(?:.\d+)?)', l)
                    if m1: abt.update(m1.groupdict())
                    continue
            if l.find('Counter readings are offset corrected') >= 0:
                abt['offsetcorrected'] = True
                readingmode = 'offsets'
                continue
            # if we reach here in 'comments' mode, this is the title line
            abt['title'] = l
            continue
        elif readingmode == 'offsets':
            m = re.findall(r'(\w+)\s(\d+(?:.\d+)?)', l)
            if m:
                abt['offsets'].update(dict(m))
                for k in abt['offsets']:
                    abt['offsets'][k] = float(abt['offsets'][k])
        elif readingmode == 'params':
            abt['params'][l.split('=')[0].strip()] = float(l.split('=')[1].strip())
        elif readingmode == 'data':
            if l.startswith('Col'):
                abt['columns'].append(l.split()[2])
            else:
                abt['data'].append([float(x) for x in l.split()])
    f.close()
    # some post-processing
    # remove common prefix from column names
    maxcolnamelen = max(len(c) for c in abt['columns'])
    for l in range(1, maxcolnamelen):
        if len(set([c[:l] for c in abt['columns']])) > 1:
            break
    abt['columns'] = [c[l - 1:] for c in abt['columns']]
    # represent data as a structured array
    dt = np.dtype(zip(abt['columns'], itertools.repeat(np.double)))
    abt['data'] = np.array(abt['data'], dtype=np.double).view(dt)
    # dates and times in datetime formats
    monthnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m, i in zip(monthnames, itertools.count(1)):
        abt['startdate'] = abt['startdate'].replace(m, str(i))
    abt['startdate'] = datetime.date(*reversed([int(x) for x in abt['startdate'].split('-')]))
    abt['starttime'] = datetime.time(*[int(x) for x in abt['starttime'].split(':')])
    abt['endtime'] = datetime.time(*[int(x) for x in abt['endtime'].split(':')])
    abt['start'] = datetime.datetime.combine(abt['startdate'], abt['starttime'])
    if abt['endtime'] <= abt['starttime']:
        abt['end'] = datetime.datetime.combine(abt['startdate'] + datetime.timedelta(1), abt['endtime'])
    else:
        abt['end'] = datetime.datetime.combine(abt['startdate'], abt['endtime'])
    del abt['starttime'];    del abt['startdate'];    del abt['endtime']
    # convert some fields to float
    for k in ['from', 'to', 'by', 'sampling']:
        if k in abt:
            abt[k] = float(abt[k])
    # change space and dash in title to underscore
    abt['title'] = abt['title'].replace('-', '_').replace(' ', '_')
    return abt
