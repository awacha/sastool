'''
Low-level I/O for 1D data
'''

import datetime
import numpy as np
import re
import itertools

from .. import misc


class SpecFileEOF(Exception):
    pass


def readspecscan(f, number=None):
    """Read the next spec scan in the file, which starts at the current position."""
    scan = None
    scannumber = None
    while True:
        l = f.readline()
        if l.startswith('#S'):
            scannumber = int(l[2:].split()[0])
            if not ((number is None) or (number == scannumber)):
                # break the loop, will skip to the next empty line after this
                # loop
                break
            if scan is None:
                scan = {}
            scan['number'] = scannumber
            scan['command'] = l[2:].split(None, 1)[1].strip()
            scan['data'] = []
        elif l.startswith('#C'):
            scan['comment'] = l[2:].strip()
        elif l.startswith('#D'):
            scan['datestring'] = l[2:].strip()
        elif l.startswith('#T'):
            scan['countingtime'] = float(l[2:].split()[0])
            scan['scantimeunits'] = l[2:].split()[1].strip()
        elif l.startswith('#M'):
            scan['countingcounts'] = float(l[2:].split()[0])
        elif l.startswith('#G'):
            if 'G' not in scan:
                scan['G'] = []
            scan['G'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#P'):
            if 'positions' not in scan:
                scan['positions'] = []
            scan['positions'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#Q'):
            pass
        elif l.startswith('#N'):
            n = [float(x) for x in l[2:].strip().split()]
            if len(n) == 1:
                scan['N'] = n[0]
            else:
                scan['N'] = n
        elif l.startswith('#L'):
            scan['Columns'] = [x.strip() for x in l[3:].split('  ')]
        elif not l:
            # end of file
            if scan is None:
                raise SpecFileEOF
            else:
                break
        elif not l.strip():
            break  # empty line, end of scan in file.
        elif l.startswith('#'):
            # ignore other lines starting with a hashmark.
            continue
        else:
            scan['data'].append(tuple(float(x) for x in l.split()))
    while l.strip():
        l = f.readline()
    if scan is not None:
        scan['data'] = np.array(
            scan['data'], dtype=list(zip(scan['Columns'], itertools.repeat(np.float))))
        return scan
    else:
        return scannumber


def readspec(filename, read_scan=None):
    """Open a SPEC file and read its content

    Inputs:

        filename: string
            the file to open

        read_scan: None, 'all' or integer
            the index of scan to be read from the file. If None, no scan should be read. If
            'all', all scans should be read. If a number, just the scan with that number
            should be read.

    Output:
        the data in the spec file in a dict.
    """
    with open(filename, 'rt') as f:
        sf = {'motors': [], 'maxscannumber': 0}
        sf['originalfilename'] = filename
        lastscannumber = None
        while True:
            l = f.readline()
            if l.startswith('#F'):
                sf['filename'] = l[2:].strip()
            elif l.startswith('#E'):
                sf['epoch'] = int(l[2:].strip())
                sf['datetime'] = datetime.datetime.fromtimestamp(sf['epoch'])
            elif l.startswith('#D'):
                sf['datestring'] = l[2:].strip()
            elif l.startswith('#C'):
                sf['comment'] = l[2:].strip()
            elif l.startswith('#O'):
                try:
                    l = l.split(None, 1)[1]
                except IndexError:
                    continue
                if 'motors' not in list(sf.keys()):
                    sf['motors'] = []
                sf['motors'].extend([x.strip() for x in l.split('  ')])
            elif not l.strip():
                # empty line, signifies the end of the header part. The next
                # line will be a scan.
                break
        sf['scans'] = {}
        if read_scan is not None:
            if read_scan == 'all':
                nr = None
            else:
                nr = read_scan
            try:
                while True:
                    s = readspecscan(f, nr)
                    if isinstance(s, dict):
                        sf['scans'][s['number']] = s
                        if nr is not None:
                            break
                        sf['maxscannumber'] = max(
                            sf['maxscannumber'], s['number'])
                    elif s is not None:
                        sf['maxscannumber'] = max(sf['maxscannumber'], s)
            except SpecFileEOF:
                pass
        else:
            while True:
                l = f.readline()
                if not l:
                    break
                if l.startswith('#S'):
                    n = int(l[2:].split()[0])
                    sf['maxscannumber'] = max(sf['maxscannumber'], n)
        for n in sf['scans']:
            s = sf['scans'][n]
            s['motors'] = sf['motors']
            if 'comment' not in s:
                s['comment'] = sf['comment']
            if 'positions' not in s:
                s['positions'] = [None] * len(sf['motors'])
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
    abt = {'offsetcorrected': False, 'params': {}, 'columns': [], 'data': [], 'title': '<no_title>',
           'offsets': {}, 'filename': filename}
    readingmode = ''
    for l in f:
        l = l.strip()
        if l.startswith('!') or len(l) == 0:
            continue
        elif l.startswith('%c'):
            readingmode = 'comments'
        elif l.startswith('%p'):
            readingmode = 'params'
        elif l.startswith('%d'):
            readingmode = 'data'
        elif readingmode == 'comments':
            m = re.match(
                r'(?P<scantype>\w+)-Scan started at (?P<startdate>\d+-\w+-\d+) (?P<starttime>\d+:\d+:\d+), ended (?P<endtime>\d+:\d+:\d+)', l)
            if m:
                abt.update(m.groupdict())
                continue
            else:
                m = re.match(r'Name: (?P<name>\w+)', l)
                if m:
                    abt.update(m.groupdict())
                    m1 = re.search(r'from (?P<from>\d+(?:.\d+)?)', l)
                    if m1:
                        abt.update(m1.groupdict())
                    m1 = re.search(r'to (?P<to>\d+(?:.\d+)?)', l)
                    if m1:
                        abt.update(m1.groupdict())
                    m1 = re.search(r'by (?P<by>\d+(?:.\d+)?)', l)
                    if m1:
                        abt.update(m1.groupdict())
                    m1 = re.search(r'sampling (?P<sampling>\d+(?:.\d+)?)', l)
                    if m1:
                        abt.update(m1.groupdict())
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
            abt['params'][l.split('=')[0].strip()] = float(
                l.split('=')[1].strip())
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
    dt = np.dtype(list(zip(abt['columns'], itertools.repeat(np.double))))
    abt['data'] = np.array(abt['data'], dtype=np.double).view(dt)
    # dates and times in datetime formats
    monthnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                  'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m, i in zip(monthnames, itertools.count(1)):
        abt['startdate'] = abt['startdate'].replace(m, str(i))
    abt['startdate'] = datetime.date(
        *reversed([int(x) for x in abt['startdate'].split('-')]))
    abt['starttime'] = datetime.time(
        *[int(x) for x in abt['starttime'].split(':')])
    abt['endtime'] = datetime.time(
        *[int(x) for x in abt['endtime'].split(':')])
    abt['start'] = datetime.datetime.combine(
        abt['startdate'], abt['starttime'])
    if abt['endtime'] <= abt['starttime']:
        abt['end'] = datetime.datetime.combine(
            abt['startdate'] + datetime.timedelta(1), abt['endtime'])
    else:
        abt['end'] = datetime.datetime.combine(
            abt['startdate'], abt['endtime'])
    del abt['starttime']
    del abt['startdate']
    del abt['endtime']
    # convert some fields to float
    for k in ['from', 'to', 'by', 'sampling']:
        if k in abt:
            abt[k] = float(abt[k])
    # change space and dash in title to underscore
    abt['title'] = abt['title'].replace('-', '_').replace(' ', '_')
    return abt
