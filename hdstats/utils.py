# Author: Dale Roberts <dale.o.roberts@gmail.com>
#
# License: BSD 3 clause

import os

def get_max_threads():
    n = os.cpu_count() or 1
    return n
