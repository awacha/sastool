#!/usb/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_lib, get_python_inc
import os

VERSION='0.0.1'

#Cython autobuilding needs the numpy headers. On Windows hosts, this trick is
# needed. On Linux, the headers are already in standard places.
incdirs=list(set([get_python_lib(0,0), get_python_lib(0,1), get_python_lib(1,0),
                  get_python_lib(1,1), get_python_inc(0), get_python_inc(1)]))
npy_incdirs=[os.path.join(x,'numpy/core/include') for x in incdirs]
incdirs.extend(npy_incdirs)

#Update the version string in the main __init__.py file.
try:
    f=open('sastool/__init__.py','rt')
    lines=f.readlines()
    f.close()
    verline=[l for l in lines if l.strip().startswith('VERSION')][0]
    verline=verline.split('=')[1].strip()[1:-1]
    if verline==VERSION:
        raise RuntimeError # to quit this try block
    f1=open('src/__init__.py','w+t')
    for l in lines:
        if l.strip().startswith('VERSION'):
            l='VERSION="%s"\n' % VERSION
        f1.write(l)
    f1.close() 
    print ""
    print "+---------------------------------------%s------------+" % ('-'*len(VERSION))
    print "| UPDATED VERSION IN src/__init__.py to %s !!!!!!!!!! |" % VERSION
    print "+---------------------------------------%s------------+" % ('-'*len(VERSION))
    print ""
except IOError:
    print "Cannot update VERSION in src/__init__.py"
except RuntimeError: 
    #this is raised when the version string is already up-to-date
    pass


#Extension modules written in Cython
ext_modules = [Extension("sastool.io._io", ["sastool/io/_io.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.sim._sim",["sastool/sim/_sim.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.utils2d._integrate", 
                         ["sastool/utils2d/_integrate.pyx"],
                         include_dirs=incdirs),
               Extension("sastool.fitting._fitfunction",
                         ["sastool/fitting/_fitfunction.pyx"],
                         include_dirs=incdirs),
               ]

setup(name='sastool', version=VERSION, author='Andras Wacha',
      author_email='awacha@gmail.com', url='http://github.com/awacha/sastool',
      description='Python macros for (A)SAXS data processing, fitting, plotting etc.',
      packages=['sastool','sastool.io','sastool.dataset','sastool.utils2d',
                'sastool.fitting','sastool.gui','sastool.sim'],
      scripts=['bin/sas2dgui.py'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      )
