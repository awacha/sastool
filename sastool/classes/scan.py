import numpy as np
import collections
import datetime
import dateutil.parser
import collections

from ..io import onedim
from .curve import GeneralCurve
import re
import time
import os

__all__ = ['ScanCurve', 'SASScan', 'read_from_spec', 'read_from_abtfio']

class ScanCurve(GeneralCurve):
    pass

class SASScan(object):
    COUNTING_MONITOR = 'MONITOR'
    COUNTING_TIME = 'TIME'
    def __init__(self, datadefs, N=10, Nincrement=10):
        """Create a new scan instance.
        
        Inputs:
            datadefs: list of tuples
                Each element is of the form ('column_name', datatype) where
                the data type is something that numpy understands (e.g. int,
                float, long or '<i8' etc. Or just a list of column names, in
                which case all columns will be assumed to be floats.
            N: the number of rows to initialize the scan with.
            Nincrement: the amount of rows to increment the space with if needed.
        """
        if isinstance(datadefs, SASScan):
            self._data = datadefs._data.copy()
            self._idx = datadefs._idx
            self.increment = datadefs.increment
            self.motors = datadefs.motors[:]
            self.motorpos = datadefs.motorpos[:]
            self.countingtype = datadefs.countingtype
            self.countingvalue = datadefs.countingvalue
            self.fsn = datadefs.fsn
            self.comment = datadefs.comment
            self.command = datadefs.command
            self.timestamp = datadefs.timestamp
            self.default_x = datadefs.default_x
            self.default_y = datadefs.default_y
            self.default_moni = datadefs.default_moni
            self._record_mode = datadefs._record_mode
            self.filename = datadefs.filename
        else:
            if all(isinstance(x, basestring) for x in datadefs):
                datadefs = [(x, float) for x in datadefs]
            self._data = np.zeros(N, dtype=datadefs)
            self._idx = 0
            self.increment = Nincrement
            self.motors = []
            self.motorpos = []
            self.countingtype = 'TIME'  # or 'MONITOR'
            self.countingvalue = 0
            self.fsn = 0
            self.comment = ''
            self.command = ''
            self.timestamp = datetime.datetime.now()
            self.default_x = 0
            self.default_y = -1
            self.default_moni = -2
            self._record_mode = False
            self.filename = None
        self._dtype = self._data.dtype
    @property
    def data(self):
        """The scan data as a structured numpy array"""
        return self._data[0:self._idx]
    @data.setter
    def data(self, newdata):
        self._data = np.array(newdata, dtype=self._dtype)
        self._idx = len(self._data)
    @property
    def dtype(self):
        """The numpy data type"""
        return self._data.dtype
    def start_record_mode(self, command, N, filename=None):
        self.command = command
        if filename is not None:
            self.filename = filename
        if self.filename is None:
            return
        self._record_mode = True
        if not os.path.exists(self.filename):
            init_spec_file(self.filename, self.comment, self.motors)
        write_scan_header_to_spec(self.filename, self, N=N)
    def stop_record_mode(self):
        self._record_mode = False
    def _convert_data(self, newdata):
        """Convert newdata to the format of the scan."""
        if not (isinstance(newdata, collections.Sequence) or isinstance(x, np.ndarray)):
            raise ValueError('Only a value of an iterable type (e.g. list, tuple, np.ndarray) can be appended')
        if len(newdata) == 0:
            return  # do nothing
        iterableelements = [(isinstance(x, collections.Sequence) or isinstance(x, np.ndarray)) and not isinstance(x, basestring) for x in newdata]
        if any(iterableelements) and not all(iterableelements):
            raise ValueError('Either all or none of the elements in newdata should be iterable.')
        if not any(iterableelements):
            newdata = [newdata]
        return np.array(newdata, dtype=self.dtype)
    def append(self, newdata):
        """Append new data to the scan."""
        newdata = self._convert_data(newdata)
        if self.get_free_space() < len(newdata):
            self.add_more_space(max(len(newdata) - self.get_free_space(), self.increment))
        self._data[self._idx:self._idx + len(newdata)] = newdata
        self._idx += len(newdata)
        write_to_spec(self.filename, newdata)
            
    def prepend(self, newdata):
        """Prepend new data to the scan."""
        if self._record_mode:
            raise ValueError('Data can only be appended in recording mode!')
        newdata = self._convert_data(newdata)
        self._data = np.hstack((newdata, self._data))
        self._idx += len(newdata)
    def get_column(self, name):
        """Return the column denoted by 'name' as a numpy vector. Same as scan[name]."""
        return self.data[name]
    def __len__(self):
        return self.data.shape    
    def get_free_space(self):
        """Get the number of the available allocated but unoccupied rows."""
        return len(self._data) - self._idx
    def add_more_space(self, N=None):
        """Allocate more rows."""
        if N is None:
            N = self.increment
        print "Extending space by ", N
        assert N > 0
        self._data = np.hstack((self._data, np.zeros(N, dtype=self.dtype)))
    def columns(self):
        """Get the names of the columns, in the same order as they appear in the
        underlying numpy array."""
        return self._data.dtype.names
    def __getitem__(self, columnname):
        if isinstance(columnname, basestring):
            return self.data[columnname]
        else:
            return self.data[self.dtype.names[columnname]]
    def get_curve(self, x=None, y=None, moni=None, scalebymonitor=True):
        """Return a ScanCurve instance.
        
        Inputs:
            x: column name or index of the abscissa
            y: column name or index of the ordinate
            moni: column name or index of the monitor column. Can be None.
            scalebymonitor: if the ordinate should be divided by monitor.
        """
        if x is None: x = self.default_x
        if y is None: y = self.default_y
        if moni is None: moni = self.default_moni
        if not isinstance(x, basestring): x = self.dtype.names[x]
        if not isinstance(y, basestring): y = self.dtype.names[y]
        if not isinstance(moni, basestring): moni = self.dtype.names[moni]
        data = self.data
        curvedata = {'x':data[x], 'y':data[y]}
        special_names = {'x':x, 'y':y, 'dx':x + 'Error', 'dy':y + 'Error'}
        if moni is not None:
            curvedata['monitor'] = data[moni]
            special_names['monitor'] = moni
            if scalebymonitor:
                curvedata['y'] = curvedata['y'] / curvedata['monitor']
        return ScanCurve(curvedata, special_names=special_names)
    
def read_from_spec(specfilename, idx):
    spec = onedim.readspec(specfilename)
    scan = spec['scans'][idx - 1]
    scn = SASScan(scan['data'].dtype, N=len(scan['data']))
    scn.data = scan['data']
    scn.motors = spec['motors']
    scn.motorpos = scan['positions']
    scn.command = scan['command']
    scn.fsn = scan['number']
    if 'countingtime' in scan:
        scn.countingvalue = scan['countingtime']
        scn.countingtype = SASScan.COUNTING_TIME
    elif 'countingcounts' in scan:
        scn.countingvalue = scan['countingcounts']
        scn.countingtype = SASScan.COUNTING_MONITOR
    scn.timestamp = float(dateutil.parser.parse(scan['datestring']).strftime('%s.%f'))
    scn.default_x = 0
    scn.default_y = -1
    scn.default_moni = -2
    scn.comment = spec['comment']
    return scn
    
def read_from_abtfio(abtfilename):
    abt = onedim.readabt(abtfilename)
    scn = SASScan(abt['data'].dtype, N=len(abt['data']))
    scn.data = abt['data']
    scn.motors = sorted(abt['params'].keys())
    scn.motorpos = [abt['params'][k] for k in scn.motors]
    scn.countingvalue = abt['sampling']
    scn.countingtype = SASScan.COUNTING_TIME
    scn.fsn = re.search('\d+', abt['name'])
    if scn.fsn is not None:
        scn.fsn = int(scn.fsn)
    scn.command = 'ABT'
    scn.comment = abt['title']
    if abt['scantype'] in ['EXAFS', 'ENERGY']:
        scn.default_x = abt['scantype']
        scn.default_y = 'TX_MUD'
        scn.default_moni = None
    else:
        scn.default_x = 0
        scn.default_y = -1
        scn.default_moni = -2
    scn.timestamp = float(abt['start'].strftime('%s.%f'))
    return scn

def init_spec_file(specfile, comment, motors):
    with open(specfile, 'w') as sf:
        sf.write('#F ' + specfile + '\n')
        dt = datetime.datetime.now()
        sf.write('#E ' + dt.strftime('%s') + '\n')
        sf.write('#D ' + dt.strftime('%a %b %d %H:%M:%S %Y') + '\n')
        sf.write('#C ' + comment + '\n')
        for i in range(len(motors) / 8 + 1):
            sf.write('#O%d ' % i + ' '.join('%8s' % m for m in motors[i * 8:(i + 1) * 8]) + '\n')

def write_scan_header_to_spec(specfile, scn, N=None):
    if not os.path.exists(specfile):
        raise IOError('File ' + specfile + ' does not exist!')
    if scn.fsn is None:
        maxfsn = 0
        with open(specfile, 'r') as sf:
            for l in specfile:
                if l.startswith('#S') and int(l.split()[1]) > maxfsn:
                    maxfsn = int(l.split()[1])
        scn.fsn = maxfsn + 1
    with open(specfile, 'a') as sf:
        sf.write('\n#S ' + str(scn.fsn) + '  ' + scn.command + '\n')
        sf.write('#D ' + datetime.datetime.fromtimestamp(scn.timestamp).strftime('%a %b %d %H:%M:%S %Y') + '\n')
        if scn.countingtype == SASScan.COUNTING_TIME:
            sf.write('#T ' + int(scn.countingvalue) + '  (Seconds)\n')
        else:
            sf.write('#M ' + int(scn.countingvalue) + '  (Counts)\n')  # I am not sure of this.
        sf.write('#G0 0\n')
        sf.write('#G1 0\n')
        sf.write('#Q 0 0 0\n')
        for i in range(len(scn.motors)):
            sf.write('#P%d ' % i + ' '.join(str(m) for m in scn.motorpos[i * 8:(i + 1) * 8]) + '\n')
        if N is None:
            N = len(scn)
        sf.write('#N ' + str(N) + '\n')
        sf.write('#L ' + '  '.join(scn.columns()) + '\n')
    
def write_to_spec(specfile, data):
    with open(specfile, 'a') as sf:
        np.savetxt(sf, data, fmt='%g', delimiter=' ', newline='\x0a')

        
