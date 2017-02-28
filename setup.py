#!/usb/bin/env python

import os
import warnings

import numpy as np
from setuptools import find_packages, setup
from setuptools.extension import Extension

# Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs = [np.get_include()]

# Extension modules written in Cython

try:
    from Cython.Build import cythonize

    pyxfiles = []
    for dir_, subdirs, files in os.walk('sastool'):
        pyxfiles.extend([os.path.join(dir_, f) for f in files if f.endswith('.pyx')])

    ext_modules = cythonize([
                                Extension(p.replace(os.path.sep, '.')[:-4], [p], include_dirs=incdirs) for p in pyxfiles
                                ])
except ImportError:
    warnings.warn('Cannot import Cython, using packaged C files instead')
    cfiles = []
    for dir_, subdirs, files in os.walk('sastool'):
        cfiles.extend([os.path.join(dir_, f) for f in files if f.endswith('.c')])
    ext_modules = [Extension(p.replace(os.path.sep, '.')[:-2], [p], include_dirs=incdirs) for p in cfiles]

setup(
    name='sastool', author='Andras Wacha',
    author_email='awacha@gmail.com', url='http://github.com/awacha/sastool',
    description='Python macros for [A]SA(X|N)S data processing, fitting, plotting etc.',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['numpy>=1.0.0', 'scipy>=0.7.0', 'matplotlib',
                      'h5py>=2.0', 'xlrd', 'xlwt'],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    keywords="saxs sans sas small-angle scattering x-ray neutron",
    license="BSD 3-clause",
    zip_safe=False,
    classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English', 
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3',
    ]
)
