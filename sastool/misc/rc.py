import os
import re
from . import utils

__all__ = ['sastoolrc']

class SASToolRC(object):
    defaults = {'gui.sasimagegui.file.fileformat_B1_org':'org_%05d.cbf',
              'gui.sasimagegui.file.headerformat_B1_org':'org_%05d.header',
              'gui.sasimagegui.file.headerformat_B1_int2dnorm':'intnorm%d.log',
              'gui.sasimagegui.file.fileformat_B1_int2dnorm':'int2dnorm%d.mat',
              'gui.sasimagegui.file.fileformat_BDF':'s%07d_001.bdf',
              'gui.sasimagegui.file.headerformat_BDF':'s%07d_001.bhf',
              'gui.sasimagegui.file.fileformat_PAXE':'XE%04d.DAT',
              'gui.sasimagegui.file.headerformat_PAXE':'XE%04.DAT',
              'gui.sasimagegui.plot.palette':'jet',
              'misc.searchpath':['.'],
              }
    def __init__(self, configfile=os.path.expanduser('~/.sastoolrc')):
        self.configfile = configfile
    def get(self, path):
        if not os.path.exists(self.configfile):
            f = open(self.configfile, 'w+t')
            f.close()
        with open(self.configfile, 'rt') as f:
            for l in f:
                l = l.strip()
                m = re.match('^' + path.replace('.', '\.') + '\s*[:=]\s*(.*)$', l)
                if m is None:
                    continue
                rhs = m.group(1).strip()
                value = utils.parse_number(rhs)
                return value
        if path in self.defaults:
            value = self.defaults[path]
            self.set(path, value)
            return value
        else:
            raise KeyError('Not found in rc: ' + path)
    def set(self, path, value):
        if os.path.exists(self.configfile):
            with open(self.configfile, 'rt') as f:
                lines = f.readlines()
        else:
            lines = []
        found = False
        for i in range(len(lines)):
            l = lines[i].strip()
            if re.match('^' + path.replace('.', '\.'), l) is None:
                continue
            lines[i] = path + ' : ' + str(value)
            found = True
            break
        if not found:
            lines.append(path + ' : ' + str(value))
        with open(self.configfile, 'wt') as f:
            for l in sorted(lines):
                if l.strip():
                    f.write(l.strip() + '\n')
    def __getitem__(self, key):
        return self.get(key)
    def __setitem__(self, key, value):
        return self.set(key, value)

sastoolrc = SASToolRC()
