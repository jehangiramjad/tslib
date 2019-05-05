from distutils.core import setup
from Cython.Build import cythonize
 
setup(
    ext_modules = cythonize("TSmodelc.pyx")
)
setup(
    ext_modules = cythonize("tsSVDModelc.pyx")
)
