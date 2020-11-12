"""
hdstats: High-dimensional statistics.
"""

import sys
from setuptools import setup, find_packages, Extension

try:
    import numpy as np
    include_dirs = [np.get_include()]
except ImportError:
    include_dirs = []

# macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
macros = []

if sys.platform == 'darwin':
    # Needs openmp lib installed: brew install libomp
    cc_flags = ["-I/usr/local/include", "-Xpreprocessor", "-fopenmp"]
    ld_flags = ["-L/usr/local/lib", "-lomp"]
else:
    cc_flags = ['-fopenmp']
    ld_flags = ['-fopenmp']

build_cfg = dict(
    include_dirs=include_dirs,
    extra_compile_args=cc_flags,
    extra_link_args=ld_flags,
    define_macros=macros,
)
print(build_cfg)

extensions = [
    Extension('hdstats.geomedian', ['hdstats/geomedian.pyx'], **build_cfg),
    Extension('hdstats.pcm', ['hdstats/pcm.pyx'], **build_cfg),
    Extension('hdstats.ts', ['hdstats/ts.pyx'], **build_cfg),
    Extension('hdstats.dtw', ['hdstats/dtw.pyx'], **build_cfg),
    Extension('hdstats.truncated', ['hdstats/truncated.pyx', 'hdstats/randomkit.c'],
              depends=['hdstats/randomkit.h'], **build_cfg),
    Extension('hdstats.wishart', ['hdstats/wishart.pyx', 'hdstats/randomkit.c'],
              depends=['hdstats/randomkit.h'], **build_cfg),
]

tests_require = [
    "pytest",
    "joblib",
]

setup(
    name="hdstats",
    packages=find_packages(".", exclude=['tests']),
    include_package_data=True,
    package_data={'': ['hdstats/*.pyx', 'hdstats/*.pyx', 'hdstats/*.h', 'hdstats/*.c']},
    setup_requires=["Cython>=0.23", "numpy", "scipy"],
    install_requires=["numpy", "scipy", "astropy"],
    tests_require=tests_require,
    extras_require={
        'test': tests_require,
    },
    version="0.1.7.post5",
    description="High-dimensional statistics.",
    url="http://github.com/daleroberts/hdstats",
    author="Dale Roberts",
    author_email="dale.o.roberts@gmail.com",
    license="Apache 2.0",
    zip_safe=False,
    ext_modules=extensions
)
