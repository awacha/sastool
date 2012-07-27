'''
Created on Jul 25, 2012

@author: andris
'''
import os
import sys
from utils import normalize_listargument
from searchpath import get_search_path

def find_subdirs(startdir = '.', recursion_depth = None):
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
    direct_subdirs = [os.path.join(startdir, x) for x in os.listdir(startdir) if os.path.isdir(os.path.join(startdir, x))]
    if recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = recursion_depth - 1
    if recursion_depth <= 1:
        return [startdir] + direct_subdirs
    else:
        subdirs = []
        for d in direct_subdirs:
            subdirs.extend(find_subdirs(d, next_recursion_depth))
        return [startdir] + subdirs

def findfileindirs(filename, dirs = None, use_pythonpath = True, use_searchpath = True, notfound_is_fatal = True, notfound_val = None):
    """Find file in multiple directories.
    
    Inputs:
        filename: the file name to be searched for.
        dirs: list of folders or None
        use_pythonpath: use the Python module search path
        use_searchpath: use the sastool search path.
        notfound_is_fatal: if an exception is to be raised if the file cannot be
            found. 
        notfound_val: the value which should be returned if the file is not
            found (only relevant if notfound_is_fatal is False)
    
    Outputs: the full path of the file.
    
    Notes:
        if filename is an absolute path by itself, folders in 'dir' won't be
            checked, only the existence of the file will be verified.
    """
    if os.path.isabs(filename):
        if os.path.exists(filename):
            return filename
        elif notfound_is_fatal:
            raise IOError('File ' + filename + ' not found.')
        else:
            return notfound_val
    if dirs is None:
        dirs = []
    dirs = normalize_listargument(dirs)
    if not dirs: #dirs is empty
        dirs = ['.']
    if use_pythonpath:
        dirs.extend(sys.path)
    if use_searchpath:
        dirs.extend(get_search_path())
    #expand ~ and ~user constructs
    dirs = [os.path.expanduser(d) for d in dirs]
    for d in dirs:
        if os.path.exists(os.path.join(d, filename)):
            return os.path.join(d, filename)
    if notfound_is_fatal:
        raise IOError('File %s not found in any of the directories.' % filename)
    else:
        return notfound_val
