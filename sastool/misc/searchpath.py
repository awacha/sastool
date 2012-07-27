'''
Created on Jul 25, 2012

@author: andris
'''
import os
import collections
import sys

class SearchPath(object):
    def __init__(self, path = None):
        self._path = []
        if isinstance(path, collections.Sequence):
            for p in path:
                self.append(p)
        elif isinstance(path, basestring):
            self.append(p)
        elif path is None:
            pass
        else:
            raise ValueError('Invalid initialization of SearchPath with type %s' % repr(type(path)))
    def append(self, *args):
        for path in args:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(path):
                raise IOError('Nonexistent path: ' + path)
            if os.path.isdir(path):
                self._path.append(os.path.abspath(os.path.expanduser(path)))
            else:
                raise ValueError(path + ' is not a directory')
    def get(self):
        return self._path[:]
    def set(self, pathlist):
        oldpath = self._path
        try:
            self._path = []
            for p in pathlist:
                self.append(p)
        except (IOError, ValueError):
            self._path = oldpath
            raise
    def add_python_path(self):
        for p in sys.path:
            self.append(p)
    def remove(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if path in self._path:
            self._path.remove(path)
        else:
            raise ValueError(path + ' not in path list')
    def remove_duplicates(self):
        path1 = []
        for p in self._path:
            if p not in path1:
                path1.append(p)
        self._path = path1

sastool_search_path = SearchPath('.')

append_search_path = sastool_search_path.append
get_search_path = sastool_search_path.get
remove_from_search_path = sastool_search_path.remove
set_search_path = sastool_search_path.set
