import datetime
import warnings
import weakref

import dateutil.parser
import numpy as np


class SpecScan(object):
    def __init__(self, specfile, index, linenumber, command, startdate=None, comment='', motor=''):
        self.specfile = specfile
        self.index = index
        self.linenumber = linenumber
        self.command = command
        self.motor = motor
        if startdate is None:
            startdate = datetime.datetime.now()
        self.startdate = startdate
        self.comment = comment
        self.data = None
        self.motorpositionsatstart = []
        self.columnnames = []

    def reload(self):
        with open(self.specfile.filename, 'rt', encoding='utf-8') as f:
            for i in range(self.linenumber):
                f.readline()
            l = f.readline()  # this is the #S line
            begin, index, command = l.split(None, 2)
            assert begin == '#S'
            assert int(index) == self.index
            assert command == self.command
            while l.startswith('#'):
                # read the header
                l = f.readline()
                if not l:
                    # EOF, happens when no scan data is recorded.
                    break
                elif not l.startswith('#'):
                    # this line is the first data line
                    break
                elif l.startswith('#D '):
                    self.startdate = dateutil.parser.parse(l.split(None, 1)[1])
                elif l.startswith('#C '):
                    self.comment = l.split(None, 1)[1]
                elif l.startswith('#T '):
                    self.countingtime = float(l.split(None, 2)[1])
                    self.countingunits = l.strip().split(None, 2)[2][1:-1]
                elif l.startswith('#P'):
                    self.motorpositionsatstart.extend([float(x) for x in l.split()[1:]])
                elif l.startswith('#N '):
                    self.expectedlength = int(l.split(None, 1)[1])
                elif l.startswith('#L '):
                    self.columnnames = l.split(None)[1:]
                else:
                    pass
                    # warnings.warn('Unknown line in scan #{:d} of scanfile {}: {}'.format(self.index, self.specfile.filename, l))
            self.dtype = np.dtype(list(zip(self.columnnames, [np.float] * len(self.columnnames))))
            if not l:
                # no data in the file
                self.data = np.zeros(0, dtype=self.dtype)
            else:
                self.data = []
                while l.strip():
                    self.data.append(tuple([float(x) for x in l.split()]))
                    l = f.readline()
                self.data = np.array(self.data, dtype=self.dtype)


class SpecFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.toc = {}
        self.reload()

    def get_scan(self, index):
        return self.toc[index]

    def max_scans(self):
        return max(self.toc.keys())

    def reload(self):
        with open(self.filename, 'rt', encoding='utf-8') as f:
            l = ''
            # read the header part
            lineindex = -1
            while not l.startswith('#S '):
                l = f.readline()
                lineindex += 1
                if l.startswith('#F '):
                    self.original_filename = l.split(None, 1)[1]
                elif l.startswith('#E '):
                    self.epoch = float(l.split(None, 1)[1])
                elif l.startswith('#D '):
                    self.datecreated = dateutil.parser.parse(l.split(None, 1)[1])
                elif l.startswith('#C '):
                    self.comment = l.split(None, 1)[1]
                elif l.startswith('#S '):
                    break
                elif l.startswith('#'):
                    warnings.warn('Unknown line in the header of scan file {}: {}'.format(self.filename, l))
                elif not l.strip():
                    pass
                else:
                    assert False
            index = None
            while True:
                if l.startswith('#S '):
                    begin, index, command = l.split(None, 2)
                    index = int(index)
                    self.toc[index] = SpecScan(weakref.proxy(self), index, lineindex, command)
                elif l.startswith('#D '):
                    self.toc[index].startdate = dateutil.parser.parse(l.split(None, 1)[1])
                elif l.startswith('#C '):
                    self.toc[index].comment = l.split(None, 1)[1]
                elif l.startswith('#L '):
                    self.toc[index].motor = l.split()[1]
                else:
                    pass
                l = f.readline()
                lineindex += 1
                if not l:
                    break
