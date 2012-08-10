#This file contains the just the version string. It is open()-ed by setup.py to
# determine the current package version and import-ed by the package on runtime.

import pkg_resources

__version__ = pkg_resources.get_distribution('sastool').version
