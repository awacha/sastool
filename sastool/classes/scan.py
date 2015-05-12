import numpy as np
import collections
import datetime
import dateutil.parser
import warnings
import weakref

from ..io import onedim
from .curve import GeneralCurve
import re
import time
import os
from functools import reduce

__all__ = ['ScanCurve', 'SASScan', 'SASScanStore']


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
            N: the number of rows to initialize the scan with. Can be a tuple
                of integers as well. In this case a multi-dimensional scan
                (imaging) is considered.
            Nincrement: the amount of rows to increment the space with if
                needed.
        """
        if isinstance(datadefs, SASScan):
            self._data = datadefs._data.copy()
            self._idx = datadefs._idx
            if hasattr(datadefs, '_N'):
                self._N = datadefs._N
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
            self.scanstore = datadefs.scanstore
        else:
            if not isinstance(datadefs, np.dtype) and all(isinstance(x, str) for x in datadefs):
                datadefs = [(x, float) for x in datadefs]
            if isinstance(N, tuple):
                Nrows = reduce(lambda a, b: a * b, N)
            else:
                Nrows = N
            self._data = np.zeros(Nrows, dtype=datadefs)
            for f in self._data.dtype.names:
                self._data[f] = np.NAN
            self._N = N
            self._idx = 0
            self.increment = Nincrement
            self.motors = []
            self.motorpos = []
            self.countingtype = 'TIME'  # or 'MONITOR'
            self.countingvalue = 0
            self.fsn = 0
            self.comment = ''
            self.command = ''
            self.timestamp = float(datetime.datetime.now().strftime('%s.%f'))
            self.default_x = 0
            self.default_y = -1
            self.default_moni = -2
            self._record_mode = False
            self.scanstore = None
        self._dtype = self._data.dtype

    @property
    def data(self):
        """The scan data as a structured numpy array"""
        return self._data[0:self._idx]

    @data.setter
    def data(self, newdata):
        self._data = np.array(newdata, dtype=self.dtype)
        self._idx = len(self._data)

    @property
    def dtype(self):
        """The numpy data type"""
        return self._data.dtype

    def start_record_mode(self, command, N, scanstore=None):
        self.command = command
        if scanstore is not None:
            self.scanstore = scanstore
        if self.scanstore is None:
            return
        self._record_mode = True
        if self not in self.scanstore.scans:
            self.scanstore.add_scan(self, N)

    def stop_record_mode(self):
        self._record_mode = False

    def append(self, newdata):
        """Append new data to the scan.

        Input:
            newdata: the new data. It must either be a tuple, a list of tuples or a one or two-dimensional np.ndarray.
        """
        if isinstance(newdata, tuple):
            newdata = np.array([newdata], dtype=self.dtype)
        elif isinstance(newdata, list):
            newdata = np.array(newdata, dtype=self.dtype)
        elif isinstance(newdata, np.ndarray) and newdata.ndim == 1:
            newdata = np.array([tuple(newdata.tolist())], dtype=self.dtype)
        elif isinstance(newdata, np.ndarray) and newdata.ndim == 2:
            newdata = np.array([tuple(x)
                                for x in newdata.tolist()], dtype=self.dtype)
        else:
            raise TypeError('Invalid type for data to be appended to scan.')
        if self.get_free_space() < len(newdata):
            self.add_more_space(
                max(len(newdata) - self.get_free_space(), self.increment))
        self._data[self._idx:self._idx + len(newdata)] = newdata
        self._idx += len(newdata)
        if self.scanstore is not None and self._record_mode:
            self.scanstore().append_data(newdata)

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

    def get_image(self, name):
        return self._data[name].reshape(self._N[::-1])

    def __len__(self):
        return len(self.data)

    def get_free_space(self):
        """Get the number of the available allocated but unoccupied rows."""
        return len(self._data) - self._idx

    def add_more_space(self, N=None):
        """Allocate more rows."""
        if N is None:
            N = self.increment
        assert N > 0
        self._data = np.hstack((self._data, np.zeros(N, dtype=self.dtype)))

    def columns(self):
        """Get the names of the columns, in the same order as they appear in the
        underlying numpy array."""
        return self._data.dtype.names

    def __getitem__(self, columnname):
        if isinstance(columnname, str):
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

        curvedata = {'x': self.get_x(x), 'y': self.get_y(y)}
        special_names = {'x': self.get_dataname('x', x),
                         'y': self.get_dataname('y', y),
                         'dx': self.get_dataname('x', x) + 'Error',
                         'dy': self.get_dataname('y', y) + 'Error'}
        if moni is not None:
            curvedata['monitor'] = self.get_moni(moni)
            special_names['monitor'] = self.get_dataname('moni', moni)
            if scalebymonitor:
                curvedata['y'] = curvedata['y'] / curvedata['monitor']
        return ScanCurve(curvedata, special_names=special_names)

    def get_dataname(self, what, label=None):
        if what not in ('x', 'y', 'moni'):
            raise ValueError(
                'Argument "what" should be either "x" or "y" or "moni".')
        if label is None:
            label = getattr(self, 'default_' + what)
        if not isinstance(label, str):
            label = self.dtype.names[label]
        return label

    def get_x(self, label=None):
        return self.data[self.get_dataname('x', label)]

    def get_y(self, label=None):
        return self.data[self.get_dataname('y', label)]

    def get_moni(self, label=None):
        return self.data[self.get_dataname('moni', label)]

    def diff(self, n=1):
        copy = SASScan(self)
        copy.data = np.zeros(len(self) - n, dtype=self.data.dtype)
        for name in self.data.dtype.names:
            if name == self.get_dataname('x'):
                x = self.data[name]
                for i in range(n):
                    x = 0.5 * (x[1:] + x[:-1])
                copy.data[name] = x
            else:
                copy.data[name] = np.diff(self.data[name], n)
        copy.comment = 'Derivative (n=%d) of ' % n + self.comment
        return copy

    def integrate(self):
        copy = SASScan(self)
        copy.data = np.zeros(len(self), dtype=self.data.dtype)
        for name in self.data.dtype.names:
            if name == self.get_dataname('x'):
                x = self.data[name]
                copy.data[name] = x
            else:
                copy.data[name] = np.cumsum(self.data[name])
        copy.comment = 'Integrate of ' + self.comment
        return copy

    @classmethod
    def read_from_spec(cls, specfilename, idx=None):
        """Read a scan from a SPEC file.

        Inputs:
            specfilename: string or dict.
                the name of the spec file (if it is a string), or
                the scan loaded from the spec file as a dict

        Output:
            a SASScan object
        """
        if not isinstance(specfilename, dict):
            spec = onedim.readspec(specfilename, idx)
            scan = spec['scans'][idx]
        else:
            scan = specfilename
        scn = cls(scan['data'].dtype, N=len(scan['data']))

        scn.data = scan['data']
        scn.data.sort(order=scn.data.dtype.names[0])
        scn.motors = scan['motors']
        scn.motorpos = scan['positions']
        scn.command = scan['command']
        scn.fsn = scan['number']
        scn._N = scan['N']
        if 'countingtime' in scan:
            scn.countingvalue = scan['countingtime']
            scn.countingtype = SASScan.COUNTING_TIME
        elif 'countingcounts' in scan:
            scn.countingvalue = scan['countingcounts']
            scn.countingtype = SASScan.COUNTING_MONITOR
        scn.timestamp = float(
            dateutil.parser.parse(scan['datestring']).strftime('%s.%f'))
        scn.default_x = 0
        scn.default_y = -1
        scn.default_moni = -2
        scn.comment = scan['comment']
        return scn

    @classmethod
    def read_from_abtfio(cls, abtfilename):
        abt = onedim.readabt(abtfilename)
        scn = cls(abt['data'].dtype, N=len(abt['data']))
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


class SASScanStore(object):

    """A class for storing more scans belonging together (i.e. done by the same
    user in the same beamtime.) Currently this corresponds to a spec file, but
    is meant to extended later on. The API is not to be regarded as stable.
    """

    def __init__(self, filename, comment=None, motors=None):
        self.scans = {}
        if not os.path.exists(filename):
            self.datetime = datetime.datetime.now()
            self.epoch = int(self.datetime.strftime('%s'))
            if motors is None:
                motors = []
            self.motors = motors
            self.comment = comment
            self.maxnumber = 0
            with open(filename, 'w') as sf:
                sf.write('#F ' + filename + '\n')
                sf.write('#E ' + str(self.epoch) + '\n')
                sf.write('#D ' + self.datetime.strftime(
                    '%a %b %d %H:%M:%S %Y') + '\n')
                sf.write('#C ' + self.comment + '\n')
                for i in range(len(self.motors) / 8 + 1):
                    sf.write('#O%d ' % i + '  '.join(
                        '%8s' % m for m in self.motors[i * 8:(i + 1) * 8]) + '\n')
        else:
            spec = onedim.readspec(filename, None)
            self.epoch = spec['epoch']
            self.datetime = spec['datetime']
            self.comment = spec['comment']
            self.motors = spec['motors']
            self.maxnumber = spec['maxscannumber']
            if motors is not None and set(motors) != set(self.motors):
                warnings.warn('Different motors in SPEC file!')
            del spec
        self.filename = filename

    @property
    def nextscan(self):
        return self.maxnumber + 1

    def add_scan(self, scn, N=None):
        if scn.fsn is None:
            scn.fsn = self.nextscan
        with open(self.filename, 'a') as sf:
            sf.write('\n#S ' + str(scn.fsn) + '  ' + scn.command + '\n')
            sf.write('#D ' + datetime.datetime.fromtimestamp(
                scn.timestamp).strftime('%a %b %d %H:%M:%S %Y') + '\n')
            sf.write('#C ' + scn.comment + '\n')
            if scn.countingtype == SASScan.COUNTING_TIME:
                sf.write('#T ' + str(scn.countingvalue) + '  (Seconds)\n')
            else:
                sf.write('#M ' + str(scn.countingvalue) + '  (Counts)\n')
                # I am not sure of this.
            sf.write('#G0 0\n')
            sf.write('#G1 0\n')
            sf.write('#Q 0 0 0\n')
            for i in range(len(scn.motors)):
                sf.write('#P%d ' % i + ' '.join(str(m)
                                                for m in scn.motorpos[i * 8:(i + 1) * 8]) + '\n')
            if N is None:
                N = len(scn)
            if isinstance(N, tuple):
                sf.write('#N ' + '  '.join([str(x) for x in N]) + '\n')
            else:
                sf.write('#N ' + str(N) + '\n')
            sf.write('#L ' + '  '.join(scn.columns()) + '\n')
        self.scans[self.nextscan] = scn
        scn.scanstore = weakref.ref(self)
        self.maxnumber = self.maxnumber + 1

    def append_data(self, data):
        with open(self.filename, 'ab') as sf:
            np.savetxt(sf, data, fmt='%g', delimiter=' ', newline='\x0a')

    def get_scan(self, idx):
        if idx < 1 or idx > self.maxnumber:
            raise ValueError('Invalid scan index!')
        if idx not in self.scans:
            self.scans[idx] = SASScan.read_from_spec(self.filename, idx)
        return self.scans[idx]

    def __getitem__(self, value):
        return self.get_scan(value)

    def finalize(self):
        for scn in self.scans:
            del self.scans[scn]
        del self.scans

    def __len__(self):
        return self.maxnumber

    def __iter__(self):
        return ScanStoreIterator(self)


class ScanStoreIterator(object):

    def __init__(self, scanstore):
        self._scanstore = scanstore
        self._i = 1

    def __next__(self):
        try:
            self._i += 1
            return self._scanstore[self._i - 1]
        # invalid scan index, will be raised by self._scanstore
        except ValueError:
            raise StopIteration
