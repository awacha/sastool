#!/usb/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib, get_python_inc
import os

#Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs = list(set([get_python_lib(0, 0), get_python_lib(0, 1), get_python_lib(1, 0),
                  get_python_lib(1, 1), get_python_inc(0), get_python_inc(1)]))
npy_incdirs = [os.path.join(x, 'numpy/core/include') for x in incdirs]
incdirs.extend(npy_incdirs)

#Extension modules written in Cython
ext_modules = [Extension("sastool.io._io", ["sastool/io/_io.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.sim._sim", ["sastool/sim/_sim.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.utils2d._integrate",
                         ["sastool/utils2d/_integrate.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.fitting._fitfunction",
                         ["sastool/fitting/_fitfunction.pyx"],
                         include_dirs=incdirs),
               ]

setup(name='sastool', version='0.2.0', author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/sastool',
      description='Python macros for [A]SA(X|N)S data processing, fitting, plotting etc.',
      packages=find_packages(),
      #cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize(ext_modules),
      install_requires=['numpy>=1.0.0', 'scipy>=0.7.0', 'matplotlib',
                          'h5py>=1.2', 'xlrd', 'xlwt'],
      setup_requires=['Cython>=0.15'],
#      entry_points={'gui_scripts':['sas2dutil = sastool:_sas2dgui_main_program'],
#                    },
      keywords="saxs sans sas small-angle scattering x-ray neutron",
      license="",
      )
