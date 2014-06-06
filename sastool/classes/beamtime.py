import os
import re
import sastool

__all__ = ['SASBeamTime']
# file formats can contain a subset of C-style format strings:
#
#   %(?P<leadingzero>0?)(?P<ndigits>\d*)(?P<datatype>[dui]{1})

def _strfrommatch(m):
    if m.groupdict()['ndigits'] and m.groupdict()['leadingzero']:
        return '(?P<fsn>\d{%s})' % m.groupdict()['ndigits']
    elif m.groupdict()['ndigits']:
        return '\s*(?P<fsn>\d+)'
    else:
        return '(?P<fsn>\d+)'

def formatstring_to_regexp(formatstring):
    return re.sub(r'%(?P<leadingzero>0?)(?P<ndigits>\d*)(?P<datatype>[dui]{1})', _strfrommatch, formatstring)

    
class SASBeamTime(object):
    """A class representing a set of SAS exposures, with a given file format and a set of directories and fsns."""
    def __init__(self, path, exposureformat, headerformat=None, minfsn=None, maxfsn=None, recursive_path=False, callbackfunc=None,
                 exposure_output_path=None, header_output_path=None):
        if isinstance(path, basestring):
            path = [path]
        if not all(isinstance(p, basestring) for p in path):
            raise ValueError('Path should be either a single folder name or a list of folder names.')
        if recursive_path:
            for p in path[:]:
                path.extend(sastool.misc.find_subdirs(p, recursive_path))
        path = [os.path.realpath(p) for p in path]
        path_uniq = []
        for p in path:
            if p not in path_uniq:
                path_uniq.append(p)
        path = path_uniq
        if not all(os.path.isdir(p) for p in path):
            raise ValueError('All elements of path should be directories. The following are not: ' + ';'.join(p for p in path if not os.path.isdir(p)))
        self.path = path
        self.exposureformat = exposureformat
        if headerformat is None:
            self.headerformat = exposureformat
        else:
            self.headerformat = headerformat
        self.minfsn = minfsn
        self.maxfsn = maxfsn
        self._headercache = []
        self.callbackfunc = callbackfunc
        self._cache_headers(True)
    def _cache_headers(self, force=False):
        if force:
            self._headercache = []
        regex = re.compile(formatstring_to_regexp(self.headerformat))
        have_fsns = [h['FSN'] for h in self._headercache]
        fsns = set()
        for d in self.path:
            fsns.update(set([int(m.groupdict()['fsn']) for m in [regex.match(f) for f in os.listdir(d)] if m is not None]))
        if self.minfsn is not None:
            fsns = set(f for f in fsns if f >= self.minfsn)
        if self.maxfsn is not None:
            fsns = set(f for f in fsns if f <= self.maxfsn)
        fsns = [f for f in fsns if f not in have_fsns]
        def _load(fsn):
            if callable(self.callbackfunc):
                self.callbackfunc()
            return sastool.classes.SASHeader(self.headerformat % f, dirs=self.path)
        self._headercache = sorted(self._headercache + [_load(f) for f in fsns], key=lambda h:h['FSN'])
            
    def find(self, *args, **kwargs):
        """Find one or more headers matching given criteria.
        
        Call this function as:
        beamtime.find(<fieldname>, <value>, [<fieldname2>, <value2>, ...])

        If <value> is callable, it is called with the field value of each header. It should return
        True for matching, False for non-matching headers. Exceptions are fatal.
        
        If <value> is not callable, a simple equality test is done.
        
        Keyword arguments:
            'returnonly': if the list of results has to be narrowed down based on the experiment date.
                Can be: 'next': returns the first matching experiment which is after the given date
                        'previous': returns the most recent matching experiment before the given date
                        'nearest': returns the nearest-in-time matching experiment
                The date used for comparision is given in 'date'
            'date': the date used for comparision (this argument should always be defined when 'returnonly' is.
            'includeflagged': include exposures which are flagged as erroneous (by looking at the 'ErrorFlags'
                header field). Default: False
        """
        if not self._headercache:
            self._cache_headers()
        lis = self._headercache[:]
        if 'includeflagged' not in kwargs:
            kwargs['includeflagged'] = False
        while args:
            field = args[0]
            value = args[1]
            if callable(value):
                lis = [l for l in lis if value(l[field])]
            else:
                lis = [l for l in lis if value == l[field]]
            args = args[2:]
        if not kwargs['includeflagged']:
            lis = [l for l in lis if not l['ErrorFlags']]
        if 'returnonly' not in kwargs:
            return lis
        elif kwargs['returnonly'] == 'nearest':
            return sorted(lis, key=lambda l:abs(l['Date'] - kwargs['date']))[0]
        elif kwargs['returnonly'] == 'next':
            return sorted([l for l in lis if l['Date'] >= kwargs['date']], key=lambda l:abs(l['Date'] - kwargs['date']))[0]
        elif kwargs['returnonly'] == 'previous':
            return sorted([l for l in lis if l['Date'] <= kwargs['date']], key=lambda l:abs(l['Date'] - kwargs['date']))[0]
        else:
            raise NotImplementedError
    def load_exposure(self, header, *args, **kwargs):
        if isinstance(header, sastool.classes.SASHeader) or isinstance(header, dict):
            return sastool.classes.SASExposure(self.exposureformat % header['FSN'], *args, dirs=self.path, **kwargs)
        else:  # argument 'header' is a number
            return sastool.classes.SASExposure(self.exposureformat % header, *args, dirs=self.path, **kwargs)
    def find_exposure(self, *args, **kwargs):
        foundheaders = self.find(*args, **kwargs)
        if isinstance(foundheaders, list):
            return [self.load_exposure(header) for header in foundheaders]
        else:
            return self.load_exposure(foundheaders)
    def update_cache_up_to(self, newmaxfsn):
        if not self._headercache:
            self._cache_headers()
        maxfsn = max([h['FSN'] for h in self._headercache])
        if newmaxfsn <= maxfsn:
            return
        for f in range(maxfsn + 1, newmaxfsn + 1):
            try:
                h = sastool.classes.SASHeader(self.headerformat % f, dirs=self.path)
            except IOError:
                continue
            self._headercache.append(h)
        return
    def reload_header_for_fsn(self, fsn):
        self._headercache=[h for h in self._headercache if h['FSN']!=fsn]
        try:
            h=sastool.classes.SASHeader(self.headerformat % fsn, dirs=self.path)
        except IOError:
            pass
        else:
            self._headercache.append(h)
        return
    def __iter__(self):
        return iter(self._headercache)
    def refresh_cache(self, force=False):
        self._cache_headers(force)
