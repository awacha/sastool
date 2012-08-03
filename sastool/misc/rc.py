import os
import re
from . import utils

configfile = os.path.expanduser('~/.sastoolrc')

defaults = {'gui.sasimagegui.file.fileformat_B1_org':'org_%05d.cbf',
          'gui.sasimagegui.file.headerformat_B1_org':'org_%05d.header',
          'gui.sasimagegui.file.headerformat_B1_int2dnorm':'intnorm%d.log',
          'gui.sasimagegui.file.fileformat_B1_int2dnorm':'int2dnorm%d.mat',
          'gui.sasimagegui.file.fileformat_BDF':'s%07d_001.bdf',
          'gui.sasimagegui.file.headerformat_BDF':'s%07d_001.bhf',
          'gui.sasimagegui.file.fileformat_PAXE':'XE%04d.DAT',
          'gui.sasimagegui.file.headerformat_PAXE':'XE%04.DAT',
          'misc.searchpath':['.'],
          }

class SASToolRC(object):
    @staticmethod
    def get(path):
        if not os.path.exists(configfile):
            f = open(configfile, 'w+t')
            f.close()
        with open(configfile, 'rt') as f:
            for l in f:
                l = l.strip()
                m = re.match('^' + path.replace('.', '\.') + '\s*[:=]\s*(.*)$', l)
                if m is None:
                    continue
                rhs = m.group(1).strip()
                value = utils.parse_number(rhs)
                return value
        if path in defaults:
            value = defaults[path]
            SASToolRC.set(path, value)
            return value
        else:
            raise KeyError('Not found in rc: ' + path)
    @staticmethod
    def set(path, value):
        if os.path.exists(configfile):
            with open(configfile, 'rt') as f:
                lines = f.readlines()
        else:
            lines = []
        found = False
        for i in range(len(lines)):
            l = lines[i].strip()
            if re.match('^' + path.replace('.', '\.'), l) is None:
                continue
            lines[i] = path + ' : ' + unicode(value)
            found = True
            break
        if not found:
            lines.append(path + ' : ' + unicode(value))
        with open(configfile, 'wt') as f:
            for l in sorted(lines):
                if l.strip():
                    f.write(l.strip() + '\n')
    def __getitem__(self, key):
        return SASToolRC.get(key)
    def __setitem__(self, key, value):
        return SASToolRC.set(key, value)


sastoolrc = SASToolRC()
