#!/usb/bin/env python

import os

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

# Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs = [np.get_include()]

# Extension modules written in Cython

pyxfiles = []
for dir_, subdirs, files in os.walk('sastool'):
    pyxfiles.extend([os.path.join(dir_, f) for f in files if f.endswith('.pyx')])

ext_modules = [
    Extension(p.replace('/', '.')[:-4], [p], include_dirs=incdirs) for p in pyxfiles
    ]

setup(
    name='sastool', author='Andras Wacha',
    author_email='awacha@gmail.com', url='http://github.com/awacha/sastool',
    description='Python macros for [A]SA(X|N)S data processing, fitting, plotting etc.',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, force=True),
    install_requires=['numpy>=1.0.0', 'scipy>=0.7.0', 'matplotlib',
                      'h5py>=2.0', 'xlrd', 'xlwt', 'Cython>=0.15'],
    use_scm_version=True,
    setup_requires=['Cython>=0.15', 'setuptools_scm'],
    keywords="saxs sans sas small-angle scattering x-ray neutron",
    license="BSD 3-clause",
    zip_safe=False,
)
