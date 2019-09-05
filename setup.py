"""
hdstats: High-dimensional statistics.
"""

import numpy as np
import sys

from setuptools import setup, find_packages, Extension
from setuptools import setup, Extension

from Cython.Distutils import build_ext

#macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
macros = []

npinclude = np.get_include()
rdinclude = np.random.__path__[0] + '/'

if sys.platform == 'darwin':
    # Needs openmp lib installed: brew install libomp
    cc_flags = ["-I/usr/local/include", "-Xpreprocessor", "-fopenmp"]
    ld_flags = ["-L/usr/local/lib", "-lomp"]
else:
    cc_flags = ['-fopenmp']
    ld_flags = ['-fopenmp']

build_cfg = dict(
    include_dirs=[npinclude],
    extra_compile_args=cc_flags,
    extra_link_args=ld_flags,
    define_macros=macros,
)

extensions = [
        Extension('hdstats.geomedian', ['hdstats/geomedian.pyx'], **build_cfg),
        Extension('hdstats.pcm', ['hdstats/pcm.pyx'], **build_cfg),
        Extension('hdstats.ts', ['hdstats/ts.pyx'], **build_cfg),
        Extension('hdstats.dtw', ['hdstats/dtw.pyx'], **build_cfg),
        Extension('hdstats.truncated', ['hdstats/truncated.pyx', 'hdstats/randomkit.c'], **build_cfg),
        Extension('hdstats.wishart', ['hdstats/wishart.pyx', 'hdstats/randomkit.c'], **build_cfg),
]

setup(
    name="hdstats",
    packages=find_packages(),
    setup_requires=["pytest-runner", "Cython>=0.23"],
    install_requires=["numpy", "scipy", "astropy"],
    tests_require=["pytest", "joblib"],
    version="0.1",
    description="High-dimensional statistics.",
    url="http://github.com/daleroberts/hdstats",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    license="Apache 2.0",
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions
)
