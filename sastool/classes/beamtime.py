import os
import re
import sastool
import pickle as pickle
import time
import logging
import numpy as np
import warnings
import collections

__all__ = ['SASBeamTime']
# file formats can contain a subset of C-style format strings:
#
#   %(?P<leadingzero>0?)(?P<ndigits>\d*)(?P<datatype>[dui]{1})

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _strfrommatch(m):
    if m.groupdict()['ndigits'] and m.groupdict()['leadingzero']:
        return '(?P<fsn>\d{%s})' % m.groupdict()['ndigits']
    elif m.groupdict()['ndigits']:
        return '\s*(?P<fsn>\d+)'
    else:
        return '(?P<fsn>\d+)'

def formatstring_to_regexp(formatstring):
    return '^' + re.sub(r'%(?P<leadingzero>0?)(?P<ndigits>\d*)(?P<datatype>[dui]{1})', _strfrommatch, formatstring) + '$'


class SASBeamTime(object):
    """A class representing a set of SAS exposures, with a given file format and a set of directories and fsns."""
    def __init__(self, path, exposureformat, headerformat=None, minfsn=None, maxfsn=None, recursive_path=False, callbackfunc=None,
                 exposure_output_path=None, header_output_path=None, cachefile=None):
        self._cachefile = cachefile
        self.path = self._normalize_path(path, recursive_path)
        self.exposureformat = exposureformat
        if headerformat is None:
            self.headerformat = exposureformat
        else:
            self.headerformat = headerformat
        if minfsn is None: minfsn = -np.inf
        self.minfsn = minfsn
        if maxfsn is None: maxfsn = np.inf
        self.maxfsn = maxfsn
        self._headercache = {}
        self.callbackfunc = callbackfunc
        self._load_cache()
    def _load_cache(self):
        try:
            t0 = time.time()
            if self._cachefile:
                with open(self._cachefile, 'rt') as f:
                    self._headercache = pickle.load(f)
            logger.debug('Header cache file %s loaded in %g seconds.' % (self._cachefile, time.time() - t0))
        except Exception as exc:
            logger.warning('Could not open cache file: %s. Reason: %s' % (self._cachefile, str(exc)))
            warnings.warn('Could not open cache file: %s. Reason: %s' % (self._cachefile, str(exc)))
        self._cache_headers()
    def _normalize_path(self, path, recursive_paths=False):
        if isinstance(path, str):
            path = [path]
        if not all(isinstance(p, str) for p in path):
            raise ValueError('Path should be either a single folder name or a list of folder names.')
        # if recursive paths are desired, we add subdirectories onto the path just after their parent.
        if recursive_paths:
            path_orig = path
            path = []
            for p in path_orig[:]:
                path.extend(sastool.misc.find_subdirs(p, None))
        # prune the path from duplicate folders. Note that we avoid using set()s, because that method
        # does not retain the original ordering.
        path_orig = path
        path_unique = []
        for p in path_orig:
            if p not in path_unique:
                path_unique.append(p)
        notdirs = [p for p in path if not os.path.isdir(p)]
        if notdirs:
            raise ValueError('All elements of path should be directories. The following are not: ' + ';'.join(notdirs))
        return path
    def _cache_headers(self, force=False):
        logger.debug('Starting re-caching of headers for format: ' + self.headerformat)
        t0 = time.time()
        if force:
            self._headercache = {}
        def _load(fname):
            if isinstance(self.callbackfunc, collections.Callable):
                self.callbackfunc()
            return (sastool.classes.SASHeader(fname), os.stat(fname).st_mtime)
        regex = re.compile(formatstring_to_regexp(self.headerformat))
        have_fsns = [h[0]['FSN'] for h in self._headercache.values()]
        fsns = set()
        have_files = []
        for d in self.path:
            # find files in d, which match the regular expression and which have not yet been found in a previous directory on the path.
            # This latter criterion ensures the possibility of overriding parameter files in a folder higher up in the path, without the
            # need to touch original measurement result files.
            matches = [x for x in [(regex.match(f), f) for f in sorted(os.listdir(d))] if x[0] is not None]
            files_matched = [(f, os.stat(os.path.join(d, f)), int(m.group(1))) for (m, f) in matches if (f not in have_files)]
            # only load files which have changed since the last load or have not been loaded.
            files_to_load = [(f, stat, fsn) for (f, stat, fsn) in files_matched \
                             if (fsn <= self.maxfsn) and (fsn >= self.minfsn) and ((f not in self._headercache) or (self._headercache[f][1] < stat.st_mtime))
                             ]

            dict_for_upd = dict(list(zip([x[0] for x in files_to_load],
                                              [_load(os.path.join(d, f)) for f, s, fsn in files_to_load])))
#            print dict_for_upd
            # print dict_for_upd[0]
            self._headercache.update(dict_for_upd)
            have_files.extend(f[0] for f in files_to_load)
        t1 = time.time()
        logger.debug('Finished re-caching in %g seconds' % (t1 - t0))
        if self._cachefile is not None:
            with open(self._cachefile, 'wt') as f:
                pickle.dump(self._headercache, f)
        logger.debug('Header cache file written to %s in %g seconds.' % (self._cachefile, time.time() - t1))
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
        lis = list(sorted(self, key=lambda h:h['FSN']))
        if 'includeflagged' not in kwargs:
            kwargs['includeflagged'] = False
        while args:
            field = args[0]
            value = args[1]
            if isinstance(value, collections.Callable):
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
        if newmaxfsn > self.maxfsn:
            self.maxfsn = newmaxfsn
        max_fsn_loaded = max(h['FSN'] for h in self)
        for fsn in range (max_fsn_loaded, newmaxfsn + 1):
            self.reload_header_for_fsn(fsn)
        self.refresh_cache()
    def reload_header_for_fsn(self, fsn):
        try:
            cacheidx = [k for k in self._headercache if self._headercache[k][0]['FSN'] == fsn][0]
        except IndexError:
           # warnings.warn('FSN #%d not in cache.' % fsn)
            cacheidx = None
        try:
            filetoread = sastool.misc.findfileindirs(self.headerformat % fsn, self.path)
        except IOError:
            warnings.warn('Could not find file ' + (self.headerformat % fsn) + ' on path.')
            return
        try:
            loaded = (sastool.classes.SASHeader(filetoread), os.stat(filetoread).st_mtime)
            if cacheidx is not None:
                self._headercache[cacheidx] = loaded
            else:
                self._headercache[self.headerformat % fsn] = loaded
        except IOError:
            warnings.warn('Could not load file ' + filetoread)
            return
    def __iter__(self):
        for val in self._headercache.values():
            yield val[0]
    def refresh_cache(self, force=False):
        self._cache_headers(force)
