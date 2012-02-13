'''
Created on Jan 18, 2012

@author: andris
'''

import datetime
import numpy as np

def readspec(filename):
    f=open(filename,'rt')
    sf={}
    sf['originalfilename']=filename
    for l in f:
        if l.startswith('#F'):
            sf['filename']=l[2:].strip()
        elif l.startswith('#E'):
            sf['ordinaldate']=long(l[2:].strip())
            sf['datetime']=datetime.datetime.fromtimestamp(sf['ordinaldate'])
        elif l.startswith('#D'):
            if 'scans' in sf.keys():
                sf['scans'][-1]['datestring']=l[2:].strip()
            sf['datestring']=l[2:].strip()
        elif l.startswith('#C'):
            if 'scans' in sf.keys():
                sf['scans'][-1]['comment']=l[2:].strip()
            else:
                sf['comment']=l[2:].strip()
        elif l.startswith('#O'):
            if 'motors' not in sf.keys():
                sf['motors']=[]
            sf['motors'].extend(l.split()[1:])
        elif l.startswith('#S'):
            if 'scans' not in sf.keys():
                sf['scans']=[]
            sf['scans'].append({})
            sf['scans'][-1]['number']=long(l[2:].split()[0])
            sf['scans'][-1]['command']=l[2:].split(None,1)[1]
            sf['scans'][-1]['data']=[]
        elif l.startswith('#T'):
            sf['scans'][-1]['scantime']=float(l[2:].split()[0])
            sf['scans'][-1]['scantimeunits']=l[2:].split()[1]
        elif l.startswith('#G'):
            if 'G' not in sf['scans'][-1].keys():
                sf['scans'][-1]['G']=[]
            sf['scans'][-1]['G'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#P'):
            if 'P' not in sf['scans'][-1].keys():
                sf['scans'][-1]['P']=[]
            sf['scans'][-1]['P'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#Q'):
            pass
        elif l.startswith('#N'):
            pass
        elif l.startswith('#L'):
            sf['scans'][-1]['Columns']=l.split()[1:]
        elif len(l.strip())==0:
            pass
        elif l.startswith('#'):
            pass
        else:
            sf['scans'][-1]['data'].append([float(x) for x in l.split()])
    for s in sf['scans']:
        if 'data' in s.keys():
            s1=[tuple(d) for d in s['data']]
            s['data']=np.array(s1,dtype=zip(s['Columns'],[np.double]*len(s['Columns'])))
    return sf
