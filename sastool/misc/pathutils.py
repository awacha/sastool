'''
Created on Jul 25, 2012

@author: andris
'''
import os
import sys
from .utils import normalize_listargument
from .searchpath import sastool_search_path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ['findfileindirs']

def findfileindirs(filename, dirs=None, use_pythonpath=True, use_searchpath=True, notfound_is_fatal=True, notfound_val=None):
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
    if not dirs:  # dirs is empty
        dirs = ['.']
    if use_pythonpath:
        dirs.extend(sys.path)
    if use_searchpath:
        dirs.extend(sastool_search_path)
    # expand ~ and ~user constructs
    dirs = [os.path.expanduser(d) for d in dirs]
    logger.debug('Searching for file %s in several folders: %s' % (filename, ', '.join(dirs)))
    for d in dirs:
        if os.path.exists(os.path.join(d, filename)):
            logger.debug('Found file %s in folder %s.' % (filename, d))
            return os.path.join(d, filename)
    logger.debug('Not found file %s in any folders.' % filename)
    if notfound_is_fatal:
        raise IOError('File %s not found in any of the directories.' % filename)
    else:
        return notfound_val
