'''
Created on Jul 25, 2012

@author: andris
'''
import os
import collections
import sys
import warnings

__all__ = ['sastool_search_path', 'find_subdirs', 'append_search_path',
           'get_search_path', 'remove_from_search_path', 'set_search_path']


class SearchPath(list):

    def __init__(self, path=None):
        list.__init__(self)
        if isinstance(path, collections.Sequence):
            self.extend(path)
        elif isinstance(path, str):
            self.append(path)
        elif path is None:
            pass
        else:
            raise ValueError(
                'Invalid initialization of SearchPath with type %s' % repr(type(path)))

    def validate_path(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise IOError('Nonexistent path: ' + path)
        if os.path.isdir(path):
            return path
        else:
            raise ValueError(path + ' is not a directory')

    def append(self, path):
        super(SearchPath, self).append(self.validate_path(path))

    def append_with_subdirs(self, path, Nrecursion=None):
        path = find_subdirs(path, Nrecursion)
        super(SearchPath, self).extend([self.validate_path(p) for p in path])

    def extend(self, pathlist):
        super(SearchPath, self).extend(
            [self.validate_path(p) for p in pathlist])

    def get(self):
        return list(self)

    def set(self, pathlist):
        pathlist = [self.validate_path(p) for p in pathlist]
        self.clear()
        self.extend(pathlist)

    def clear(self):
        while len(self):
            self.pop()

    def add_python_path(self):
        for p in sys.path:
            self.append(p)

    def __setitem__(self, key, value):
        super(SearchPath, self).__setitem__(key, self.validate_path(value))

    def remove(self, path):
        print("remove called")
        path = os.path.abspath(os.path.expanduser(path))
        super(SearchPath, self).remove(path)

    def remove_duplicates(self):
        path1 = []
        for p in self[:]:
            if p not in path1:
                path1.append(p)
        self.set(path1)

sastool_search_path = SearchPath('.')


def append_search_path(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'append_search_path() is deprecated, use sastool_search_path.append() instead.'))
    return sastool_search_path.append(*args, **kwargs)


def get_search_path(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'get_search_path() is deprecated, use sastool_search_path.get() instead.'))
    return sastool_search_path.get(*args, **kwargs)


def remove_from_search_path(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'remove_from_search_path() is deprecated, use sastool_search_path.remove() instead.'))
    return sastool_search_path.remove(*args, **kwargs)


def set_search_path(*args, **kwargs):
    warnings.warn(DeprecationWarning(
        'set_search_path() is deprecated, use sastool_search_path.set() instead.'))
    return sastool_search_path.set(*args, **kwargs)


def find_subdirs(startdir='.', recursion_depth=None):
    """Find all subdirectory of a directory.

    Inputs:
        startdir: directory to start with. Defaults to the current folder.
        recursion_depth: number of levels to traverse. None is infinite.

    Output: a list of absolute names of subfolders.

    Examples:
        >>> find_subdirs('dir',0)  # returns just ['dir']

        >>> find_subdirs('dir',1)  # returns all direct (first-level) subdirs
                                   # of 'dir'.
    """
    startdir = os.path.expanduser(startdir)
    direct_subdirs = [os.path.join(startdir, x) for x in os.listdir(
        startdir) if os.path.isdir(os.path.join(startdir, x))]
    if recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = recursion_depth - 1
    if (recursion_depth is not None) and (recursion_depth <= 1):
        return [startdir] + direct_subdirs
    else:
        subdirs = []
        for d in direct_subdirs:
            subdirs.extend(find_subdirs(d, next_recursion_depth))
        return [startdir] + subdirs
