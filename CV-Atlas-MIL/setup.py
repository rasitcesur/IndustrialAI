"""Optional accelerated implementation of the original, research-grade feature operator."""
import os
import sys
import numpy as np
from setuptools import Extension,setup
from Cython.Build import cythonize

if sys.platform == 'win32':
    cargs=['/O2','/openmp'];largs=[]
else:
    cargs=['-O3','-fopenmp'];largs=['-fopenmp']

setup(name='cv-atlas-mil-kernels',ext_modules=cythonize([Extension(
    'fast_kernels',['fast_kernels.pyx'],include_dirs=[np.get_include()],
    extra_compile_args=cargs,extra_link_args=largs)],compiler_directives={'language_level':3}))
