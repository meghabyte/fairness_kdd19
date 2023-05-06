from distutils.core import setup
from Cython.Build import cythonize

setup(name='Fairness App', ext_modules=cythonize("*.pyx"))