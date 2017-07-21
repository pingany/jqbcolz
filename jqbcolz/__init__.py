########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""
jqbcolz: columnar and compressed data containers
==============================================

jqbcolz provides columnar and compressed data containers.  Column storage
allows for efficiently querying tables with a large number of columns.  It
also allows for cheap addition and removal of column.  In addition,
jqbcolz objects are compressed by default for reducing memory/disk I/O needs.
The compression process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

"""
from pkg_resources import parse_version

# Filters
NOSHUFFLE = 0
SHUFFLE = 1
BITSHUFFLE = 2

# Translation of filters to strings
filters = {NOSHUFFLE: "noshuffle",
           SHUFFLE: "shuffle",
           BITSHUFFLE: "bitshuffle"}

min_numexpr_version = '2.5.2'  # the minimum version of Numexpr needed
numexpr_here = False
try:
    import numexpr
except ImportError:
    pass
else:
    if parse_version(numexpr.__version__) >= parse_version(min_numexpr_version):
        numexpr_here = True

# Check for dask (as another virtual machine for chunked eval)
min_dask_version = '0.9.0'  # the minimum version of Numexpr needed
dask_here = False
try:
    import dask
except ImportError:
    pass
else:
    if parse_version(dask.__version__) >= parse_version(min_dask_version):
        dask_here = True

# Check for pandas (for data container conversion purposes)
pandas_here = False
try:
    import pandas
except ImportError:
    pass
else:
    pandas_here = True

# Check for PyTables (for data container conversion purposes)
tables_here = False
try:
    import tables
except ImportError:
    pass
else:
    tables_here = True

# Print array functions (imported from NumPy)
from jqbcolz.arrayprint import (
    array2string, set_printoptions, get_printoptions )

from jqbcolz.carray_ext import (
    carray, blosc_version, blosc_compressor_list,
    _blosc_set_nthreads as blosc_set_nthreads,
    _blosc_init, _blosc_destroy)
from jqbcolz.ctable import ctable
from jqbcolz.toplevel import (
    print_versions, detect_number_of_cores, set_nthreads,
    open, fromiter, arange, zeros, ones, fill,
    iterblocks, cparams, walk)
from jqbcolz.chunked_eval import eval
from jqbcolz.defaults import defaults, defaults_ctx
from jqbcolz.version import version as __version__

try:
    from jqbcolz.tests import test
except ImportError:
    def test(*args, **kwargs):
        print("Could not import tests.\n"
              "If on Python2.6 please install unittest2")


def _get_git_description(path_):
    """ Get the output of git-describe when executed in a given path. """

    # imports in function because:
    # a) easier to refactor
    # b) clear they are only used here
    import subprocess
    import os
    import os.path as path
    from jqbcolz.py2help import check_output

    # make an absolute path if required, for example when running in a clone
    if not path.isabs(path_):
        path_ = path.join(os.getcwd(), path_)
    # look up the commit using subprocess and git describe
    try:
        # redirect stderr to stdout to make sure the git error message in case
        # we are not in a git repo doesn't appear on the screen and confuse the
        # user.
        label = check_output(["git", "describe"], cwd=path_,
                             stderr=subprocess.STDOUT).strip()
        return label
    except OSError:  # in case git wasn't found
        pass
    except subprocess.CalledProcessError:  # not in git repo
        pass

git_description = _get_git_description(__path__[0])

# Initialization code for the Blosc and numexpr libraries
_blosc_init()
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
# Benchmarks show that using several threads can be an advantage in jqbcolz
blosc_set_nthreads(ncores)
if numexpr_here:
    numexpr.set_num_threads(ncores)
import atexit
atexit.register(_blosc_destroy)
