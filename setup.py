"""
hdstats: High-dimensional statistics.
"""

import numpy as np
from setuptools import setup, find_packages, Extension
from setuptools import setup, Extension

from Cython.Distutils import build_ext

macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

npinclude = np.get_include()
rdinclude = np.get_include() + '/../../random/'

extensions = [
        Extension('hdstats.geomedian', ['hdstats/geomedian.pyx'], include_dirs = [np.get_include()], define_macros=macros),
#        Extension('hdstats.wishart', ['hdstats/wishart.pyx'], include_dirs = [np.get_include(), rdinclude], define_macros=macros)
]

setup(
    name="hdstats",
    packages=find_packages(),
    setup_requires=["pytest-runner", "Cython>=0.23"],
    install_requires=["numpy", "Cython>=0.23"],
    tests_require=["pytest"],
    version="0.1",
    description="High-dimensional statistics.",
    url="http://github.com/daleroberts/hdstats",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    license="Apache 2.0",
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions
)
