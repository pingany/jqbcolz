#!python
#cython: embedsignature=True
#cython: linetrace=True
#########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted -  francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

import sys
import os
import os.path
import struct
import shutil
import tempfile
import json
import datetime
import threading
import mmap

import numpy as np
cimport numpy as np
from numpy cimport (ndarray,
                    dtype,
                    import_array,
                    PyArray_GETITEM,
                    PyArray_SETITEM,
                    npy_intp,
                    )
import cython

import jqbcolz
from jqbcolz import utils, attrs, array2string

from .utils import build_carray

if sys.version_info >= (3, 0):
    _MAXINT = 2 ** 31 - 1
    _inttypes = (int, np.integer)
else:
    _MAXINT = sys.maxint
    _inttypes = (int, long, np.integer)

_KB = 1024
_MB = 1024 * _KB

# Directories for saving the data and metadata for jqbcolz persistency
DATA_DIR = 'data'
META_DIR = 'meta'
SIZES_FILE = 'sizes'
STORAGE_FILE = 'storage'

# For the persistence layer
EXTENSION = '.blp'
MAGIC = b'blpk'
BLOSCPACK_HEADER_LENGTH = 16
BLOSC_HEADER_LENGTH = 16
FORMAT_VERSION = 1
MAX_FORMAT_VERSION = 255
MAX_CHUNKS = (2 ** 63) - 1

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = np.int64

# The native int type for this platform
IntType = np.dtype(np.int_)

#-----------------------------------------------------------------

# numpy functions & objects
from definitions cimport (malloc,
                          realloc,
                          free,
                          memcpy,
                          memset,
                          strdup,
                          strcmp,
                          PyBytes_GET_SIZE,
                          PyBytes_FromStringAndSize,
                          PyBytes_AS_STRING,
                          Py_BEGIN_ALLOW_THREADS,
                          Py_END_ALLOW_THREADS,
                          PyBuffer_FromMemory,
                          Py_uintptr_t,
                          )

#-----------------------------------------------------------------


# Check blosc version
cdef extern from "check_blosc_version.h":
    pass

# Blosc routines
cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE,
        BLOSC_MAX_TYPESIZE

    void blosc_init()
    void blosc_destroy()
    void blosc_get_versions(char *version_str, char *version_date)
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char*compname)
    int blosc_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, void *src, void *dest,
                       size_t destsize) nogil
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                           size_t nbytes, const void* src, void* dest,
                           size_t destsize, const char* compressor,
                           size_t blocksize, int numinternalthreads) nogil
    int blosc_decompress(void *src, void *dest, size_t destsize) nogil
    int blosc_decompress_ctx(const void *src, void *dest, size_t destsize,
                             int numinternalthreads) nogil
    int blosc_getitem(void *src, int start, int nitems, void *dest) nogil
    void blosc_free_resources()
    void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                             size_t *cbytes, size_t *blocksize)
    void blosc_cbuffer_metainfo(void *cbuffer, size_t *typesize, int *flags)
    void blosc_cbuffer_versions(void *cbuffer, int *version, int *versionlz)
    void blosc_set_blocksize(size_t blocksize)
    char* blosc_list_compressors()

cdef extern int PyObject_AsReadBuffer(object, const void **, Py_ssize_t *)

#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#-------------------------------------------------------------

# Some utilities
def blosc_compressor_list():
    """
    blosc_compressor_list()

    Returns a list of compressors available in the Blosc build.

    Parameters
    ----------
    None

    Returns
    -------
    out : list
        The list of names.
    """
    list_compr = blosc_list_compressors()
    if sys.version_info >= (3, 0):
        # Convert compressor names into regular strings in Python 3 (unicode)
        list_compr = list_compr.decode()
    clist = list_compr.split(',')
    return clist

def _blosc_set_nthreads(nthreads):
    """
    _blosc_set_nthreads(nthreads)

    Sets the number of threads that Blosc can use.

    Parameters
    ----------
    nthreads : int
        The desired number of threads to use.

    Returns
    -------
    out : int
        The previous setting for the number of threads.
    """
    return blosc_set_nthreads(nthreads)

def _blosc_init():
    """
    _blosc_init()

    Initialize the Blosc library.

    """
    blosc_init()

def _blosc_destroy():
    """
    _blosc_destroy()

    Finalize the Blosc library.

    """
    blosc_destroy()

def blosc_version():
    """
    blosc_version()

    Return the version of the Blosc library.

    """
    # All the 'decode' contorsions are for Python 3 returning actual strings
    ver_str = <char *> BLOSC_VERSION_STRING
    if hasattr(ver_str, "decode"):
        ver_str = ver_str.decode()
    ver_date = <char *> BLOSC_VERSION_DATE
    if hasattr(ver_date, "decode"):
        ver_date = ver_date.decode()
    return (ver_str, ver_date)

def list_bytes_to_str(lst):
    """The Python 3 JSON encoder doesn't accept 'bytes' objects,
    this utility function converts all bytes to strings.
    """
    if isinstance(lst, bytes):
        return lst.decode('ascii')
    elif isinstance(lst, list):
        return [list_bytes_to_str(x) for x in lst]
    else:
        return lst

# This is the same than in utils.py, but works faster in extensions
cdef get_len_of_range(npy_intp start, npy_intp stop, npy_intp step):
    """Get the length of a (start, stop, step) range."""
    cdef npy_intp n

    n = 0
    if start < stop:
        # Do not use a cython.cdiv here (do not ask me why!)
        n = ((stop - start - 1) // step + 1)
    return n

cdef clip_chunk(npy_intp nchunk, npy_intp chunklen,
                npy_intp start, npy_intp stop, npy_intp step):
    """Get the limits of a certain chunk based on its length."""
    cdef npy_intp startb, stopb, blen, distance

    startb = start - nchunk * chunklen
    stopb = stop - nchunk * chunklen

    # Check limits
    if (startb >= chunklen) or (stopb <= 0):
        return startb, stopb, 0  # null size
    if startb < 0:
        startb = 0
    if stopb > chunklen:
        stopb = chunklen

    # step corrections
    if step > 1:
        # Just correcting startb is enough
        distance = (nchunk * chunklen + startb) - start
        if distance % step > 0:
            startb += (step - (distance % step))
            if startb > chunklen:
                return startb, stopb, 0  # null size

    # Compute size of the clipped block
    blen = get_len_of_range(startb, stopb, step)

    return startb, stopb, blen

cdef int check_zeros(char *data, int nbytes):
    """Check whether [data, data+nbytes] is zero or not."""
    cdef int i, iszero, chunklen, leftover
    cdef size_t *sdata

    iszero = 1
    sdata = <size_t *> data
    chunklen = cython.cdiv(nbytes, sizeof(size_t))
    leftover = nbytes % sizeof(size_t)
    with nogil:
        for i from 0 <= i < chunklen:
            if sdata[i] != 0:
                iszero = 0
                break
        else:
            data += nbytes - leftover
            for i from 0 <= i < leftover:
                if data[i] != 0:
                    iszero = 0
                    break
    return iszero

cdef int true_count(char *data, int nbytes):
    """Count the number of true values in data (boolean)."""
    cdef int i, count

    with nogil:
        count = 0
        for i from 0 <= i < nbytes:
            count += <int> (data[i])
    return count

#-------------------------------------------------------------

# set the value of this variable to True or False to override the
# default adaptive behaviour
use_threads = None


def _get_use_threads():
    global use_threads

    if use_threads in [True, False]:
        # user has manually overridden the default behaviour
        _use_threads = use_threads

    else:
        # adaptive behaviour: allow blosc to use threads if it is being
        # called from the main Python thread, inferring that it is being run
        # from within a single-threaded program; otherwise do not allow
        # blosc to use threads, inferring it is being run from within a
        # multi-threaded program
        if hasattr(threading, 'main_thread'):
            _use_threads = (threading.main_thread() ==
                            threading.current_thread())
        else:
            _use_threads = threading.current_thread().name == 'MainThread'

    return _use_threads


cdef class chunk:
    """
    chunk(array, atom, cparams)

    Compressed in-memory container for a data chunk.

    This class is meant to be used only by the `carray` class.

    """

    property dtype:
        "The NumPy dtype for this chunk."
        def __get__(self):
            return self.atom

    def __cinit__(self, object dobject, object atom, object cparams,
                  object _memory=True, object _compr=False):
        cdef int itemsize, footprint
        cdef size_t nbytes, cbytes, blocksize
        cdef dtype dtype_
        cdef char *data

        self.atom = atom
        self.atomsize = atom.itemsize
        dtype_ = atom.base
        self.typekind = dtype_.kind
        # Hack for allowing strings with len > BLOSC_MAX_TYPESIZE
        if self.typekind == 'S':
            itemsize = 1
        elif self.typekind == 'U':
            itemsize = 4
        elif self.typekind == 'V' and dtype_.itemsize > BLOSC_MAX_TYPESIZE:
            itemsize = 1
        else:
            itemsize = dtype_.itemsize
        if itemsize > BLOSC_MAX_TYPESIZE:
            raise TypeError(
                "typesize is %d and jqbcolz does not currently support data "
                "types larger than %d bytes" % (itemsize, BLOSC_MAX_TYPESIZE))
        self.itemsize = itemsize
        footprint = 0

        if _compr:
            # Data comes in an already compressed state inside a Python String
            self.dobject = dobject
            # Set size info for the instance
            data = _get_object_buffer(dobject);
            blosc_cbuffer_sizes(data, &nbytes, &cbytes, &blocksize)
        elif dtype_ == 'O':
            # The objects should arrive here already pickled
            data = _get_object_buffer(dobject)
            nbytes = PyBytes_GET_SIZE(dobject)
            cbytes, blocksize = self.compress_data(data, 1, nbytes, cparams)
        else:
            # Compress the data object (a NumPy object)
            nbytes, cbytes, blocksize, footprint = self.compress_arrdata(
                dobject, itemsize, cparams, _memory)
        footprint += 128  # add the (aprox) footprint of this instance in bytes

        # Fill instance data
        self.nbytes = nbytes
        self.cbytes = cbytes
        self.blocksize = blocksize

    cdef compress_arrdata(self, ndarray array, int itemsize,
                          object cparams, object _memory):
        """Compress data in `array`"""
        cdef size_t nbytes, cbytes, blocksize, footprint

        # Compute the total number of bytes in this array
        nbytes = array.itemsize * array.size
        cbytes = 0
        footprint = 0

        # Check whether incoming data can be expressed as a constant or not.
        # Disk-based chunks are not allowed to do this.
        self.isconstant = 0
        self.constant = None
        if _memory and (array.strides[0] == 0
                        or check_zeros(array.data, nbytes)):
            self.isconstant = 1
            # Get the NumPy constant.  Avoid this NumPy quirk:
            # np.array(['1'], dtype='S3').dtype != s[0].dtype
            if array.dtype.kind != 'S':
                self.constant = array[0]
            else:
                self.constant = np.array(array[0], dtype=array.dtype)
            # Add overhead (64 bytes for the overhead of the numpy container)
            footprint += 64 + self.constant.size * self.constant.itemsize

        if self.isconstant:
            blocksize = 4 * 1024  # use 4 KB as a cache for blocks
            # Make blocksize a multiple of itemsize
            if blocksize % itemsize > 0:
                blocksize = cython.cdiv(blocksize, itemsize) * itemsize
            # Correct in case we have a large itemsize
            if blocksize == 0:
                blocksize = itemsize
        else:
            if self.typekind == 'b':
                self.true_count = true_count(array.data, nbytes)

            if array.strides[0] == 0:
                # The chunk is made of constants.  Regenerate the actual data.
                array = array.copy()

            # Quantize data if necessary before compression
            if cparams.quantize:
                array = utils.quantize(array, cparams.quantize)
            # Compress data
            cbytes, blocksize = self.compress_data(
                array.data, itemsize, nbytes, cparams)
        return (nbytes, cbytes, blocksize, footprint)

    cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                       object cparams):
        """Compress data with `cparams` and return metadata."""
        cdef size_t nbytes_, cbytes, blocksize
        cdef int clevel, shuffle, ret
        cdef char *dest
        cdef char *cname_str

        clevel = cparams.clevel
        shuffle = cparams.shuffle
        cname = cparams.cname
        if type(cname) != bytes:
            cname = cname.encode()
        if blosc_set_compressor(cname) < 0:
            raise ValueError(
                "Compressor '%s' is not available in this build" % cname)
        dest = <char *> malloc(nbytes + BLOSC_MAX_OVERHEAD)
        if _get_use_threads():
            with nogil:
                ret = blosc_compress(clevel, shuffle, itemsize, nbytes,
                                     data, dest, nbytes + BLOSC_MAX_OVERHEAD)
        else:
            cname_str = cname
            with nogil:
                ret = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes,
                                         data, dest,
                                         nbytes + BLOSC_MAX_OVERHEAD,
                                         cname_str, 0, 1)
        if ret <= 0:
            raise RuntimeError(
                "fatal error during Blosc compression: %d" % ret)
        # Copy the compressed buffer into a Bytes buffer
        cbytes = ret
        self.dobject = PyBytes_FromStringAndSize(dest, cbytes)
        # Get blocksize info for the instance
        blosc_cbuffer_sizes(dest, &nbytes_, &cbytes, &blocksize)
        assert nbytes_ == nbytes
        free(dest)

        return (cbytes, blocksize)

    def getdata(self):
        """Get a compressed string object for this chunk (for persistence)."""
        cdef object string

        assert (not self.isconstant,
                "This function can only be used for persistency")
        return self.dobject

    def getudata(self):
        """Get an uncompressed string out of this chunk (for 'O'bject types)."""
        cdef int ret
        cdef char *src
        cdef char *dest

        result_str = PyBytes_FromStringAndSize(NULL, self.nbytes)
        src = _get_object_buffer(self.dobject)
        dest = PyBytes_AS_STRING(result_str)

        if _get_use_threads():
            with nogil:
                ret = blosc_decompress(src, dest, self.nbytes)
        else:
            with nogil:
                ret = blosc_decompress_ctx(src, dest, self.nbytes, 1)
        if ret < 0:
            raise RuntimeError(
                "fatal error during Blosc decompression: %d" % ret)
        return result_str

    cdef void _getitem(self, int start, int stop, char *dest):
        """Read data from `start` to `stop` and return it as a numpy array."""
        cdef int ret, bsize, blen, nitems, nstart
        cdef ndarray constants
        cdef char *data

        blen = stop - start
        bsize = blen * self.atomsize
        nitems = cython.cdiv(bsize, self.itemsize)
        nstart = cython.cdiv(start * self.atomsize, self.itemsize)

        if self.isconstant:
            # The chunk is made of constants
            constants = np.ndarray(shape=(blen,), dtype=self.dtype,
                                   buffer=self.constant, strides=(0,)).copy()
            memcpy(dest, constants.data, bsize)
            return

        # Fill dest with uncompressed data
        data = _get_object_buffer(self.dobject)
        if bsize == self.nbytes:
            ret = blosc_decompress(data, dest, bsize)
        else:
            ret = blosc_getitem(data, nstart, nitems, dest)
        if ret < 0:
            raise RuntimeError(
                "fatal error during Blosc decompression: %d" % ret)

    def __getitem__(self, object key):
        """__getitem__(self, key) -> values."""
        cdef ndarray array
        cdef object start, stop, step, clen, idx

        if isinstance(key, _inttypes):
            # Quickly return a single element
            array = np.empty(shape=(1,), dtype=self.dtype)
            self._getitem(key, key + 1, array.data)
            return PyArray_GETITEM(array, array.data)
        elif isinstance(key, slice):
            (start, stop, step) = key.start, key.stop, key.step
        elif isinstance(key, tuple) and self.dtype.shape != ():
            # Build an array to guess indices
            clen = cython.cdiv(self.nbytes, self.itemsize)
            idx = np.arange(clen, dtype=np.int32).reshape(self.dtype.shape)
            idx2 = idx(key)
            if idx2.flags.contiguous:
                # The slice represents a contiguous slice.  Get start and stop.
                start, stop = idx2.flatten()[[0, -1]]
                step = 1
            else:
                (start, stop, step) = key[0].start, key[0].stop, key[0].step
        else:
            raise IndexError("key not suitable:", key)

        # Get the corrected values for start, stop, step
        clen = cython.cdiv(self.nbytes, self.atomsize)
        (start, stop, step) = slice(start, stop, step).indices(clen)

        # Build a numpy container
        array = np.empty(shape=(stop - start,), dtype=self.dtype)
        # Read actual data
        self._getitem(start, stop, array.data)

        # Return the value depending on the step
        if step > 1:
            return array[::step]
        return array

    def __setitem__(self, object key, object value):
        """__setitem__(self, key, value) -> None."""
        raise NotImplementedError()

    def __str__(self):
        """Represent the chunk as an string."""
        return str(self[:])

    def __repr__(self):
        """Represent the chunk as an string, with additional info."""
        cratio = self.nbytes / float(self.cbytes)
        fullrepr = "chunk(%s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
                   (self.dtype, self.nbytes, self.cbytes, cratio, str(self))
        return fullrepr


cdef create_bloscpack_header(nchunks=None, format_version=FORMAT_VERSION):
    """Create the bloscpack header string.

    Parameters
    ----------
    nchunks : int
        the number of chunks, default: None
    format_version : int
        the version format for the compressed file

    Returns
    -------
    bloscpack_header : string
        the header as string

    Notes
    -----

    The bloscpack header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | RESERVED  |           nchunks             |
                   version

    The first four are the magic string 'blpk'. The next one is an 8 bit
    unsigned little-endian integer that encodes the format version. The next
    three are reserved, and in the last eight there is a signed 64 bit little
    endian integer that encodes the number of chunks.

    Currently (jqbcolz 1.x), version is 1 and nchunks always have a value of 1
    (this might change in jqbcolz 2.0).

    The value of '-1' for 'nchunks' designates an unknown size and can be
    set by setting 'nchunks' to None.

    Raises
    ------
    ValueError
        if the nchunks argument is too large or negative
    struct.error
        if the format_version is too large or negative

    """
    if not 0 <= nchunks <= MAX_CHUNKS and nchunks is not None:
        raise ValueError(
            "'nchunks' must be in the range 0 <= n <= %d, not '%s'" %
            (MAX_CHUNKS, str(nchunks)))
    return (MAGIC + struct.pack('<B', format_version) + b'\x00\x00\x00' +
            struct.pack('<q', nchunks if nchunks is not None else -1))

if sys.version_info >= (3, 0):
    def decode_byte(byte):
        return byte
else:
    def decode_byte(byte):
        return int(byte.encode('hex'), 16)
def decode_uint32(fourbyte):
    return struct.unpack('<I', fourbyte)[0]


cdef decode_blosc_header(buffer_):
    """ Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'ctbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    return {'version': decode_byte(buffer_[0]),
            'versionlz': decode_byte(buffer_[1]),
            'flags': decode_byte(buffer_[2]),
            'typesize': decode_byte(buffer_[3]),
            'nbytes': decode_uint32(buffer_[4:8]),
            'blocksize': decode_uint32(buffer_[8:12]),
            'ctbytes': decode_uint32(buffer_[12:16])}


cdef char* _get_object_buffer(_object):
    cdef void * buf;
    cdef Py_ssize_t _len;
    cdef int offset = BLOSCPACK_HEADER_LENGTH
    if isinstance(_object, mmap.mmap):
        PyObject_AsReadBuffer(_object, &buf, &_len)
        return <char*>(buf + offset)
    else:
        return PyBytes_AS_STRING(_object)

cdef class chunks(object):
    """Store the different carray chunks in a directory on-disk."""
    property mode:
        "The mode used to create/open the `mode`."
        def __get__(self):
            return self._mode
        def __set__(self, value):
            self._mode = value

    property rootdir:
        "The on-disk directory used for persistency."
        def __get__(self):
            return self._rootdir
        def __set__(self, value):
            self._rootdir = value

    property datadir:
        """The directory for data files."""
        def __get__(self):
            return os.path.join(self.rootdir, DATA_DIR)

    def __cinit__(self, rootdir, metainfo=None, _new=False, mmap=False):
        cdef void *decompressed
        cdef void *compressed
        cdef int leftover
        cdef size_t chunksize
        cdef object scomp
        cdef int ret
        cdef int itemsize, atomsize
        cdef int chunklen

        self._rootdir = rootdir
        self._mmap = mmap
        self.nchunks = 0
        self.nchunk_cached = -1  # no chunk cached initially
        self.dtype, self.cparams, self.len, self._mode, chunklen = metainfo
        atomsize = self.dtype.itemsize
        itemsize = self.dtype.base.itemsize

        # For 'O'bject types, the number of chunks is equal to the number of
        # elements
        if self.dtype.char == 'O':
            self.nchunks = self.len

        # Initialize last chunk (not valid for 'O'bject dtypes)
        if not _new and self.dtype.char != 'O':
            self.nchunks = cython.cdiv(self.len, chunklen)
            chunksize = chunklen * atomsize
            leftover = (self.len % chunklen) * atomsize

    cdef _chunk_file_name(self, nchunk):
        """ Determine the name of a chunk file. """
        return os.path.join(self.datadir, "__%d%s" % (nchunk, EXTENSION))

    cdef read_chunk(self, nchunk):
        """Read a chunk and return it in compressed form."""
        schunkfile = self._chunk_file_name(nchunk)
        if not os.path.exists(schunkfile):
            raise ValueError("chunkfile %s not found" % schunkfile)

        if self._mmap:
            fd = os.open(schunkfile, os.O_RDONLY)
            schunk = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
            os.close(fd)
            return schunk
        else:
            with open(schunkfile, 'rb') as schunk:
                bloscpack_header = schunk.read(BLOSCPACK_HEADER_LENGTH)
                blosc_header_raw = schunk.read(BLOSC_HEADER_LENGTH)
                blosc_header = decode_blosc_header(blosc_header_raw)
                ctbytes = blosc_header['ctbytes']
                nbytes = blosc_header['nbytes']
                # seek back BLOSC_HEADER_LENGTH bytes in file relative to current
                # position
                schunk.seek(-BLOSC_HEADER_LENGTH, 1)
                scomp = schunk.read(ctbytes)
            return scomp

    def __iter__(self):
       self._iter_count = 0
       return self

    def __next__(self):
        cdef int i
        if self._iter_count < self.nchunks:
            i = self._iter_count
            self._iter_count += 1
            return self.__getitem__(i)
        else:
            raise StopIteration()

    def __getitem__(self, nchunk):
        cdef void *decompressed
        cdef void *compressed

        if nchunk == self.nchunk_cached:
            # Hit!
            return self.chunk_cached
        else:
            scomp = self.read_chunk(nchunk)
            # Data chunk should be compressed already
            chunk_ = chunk(scomp, self.dtype, self.cparams,
                           _memory=False, _compr=True)
            # Fill cache
            self.nchunk_cached = nchunk
            self.chunk_cached = chunk_
        return chunk_

    def __setitem__(self, nchunk, chunk_):
        self._save(nchunk, chunk_)

    def __len__(self):
        return self.nchunks

    def free_cachemem(self):
        self.nchunk_cached = -1
        self.chunk_cached = None


cdef class carray:
    """
    carray(array, cparams=None, dtype=None, dflt=None, expectedlen=None,
    chunklen=None, rootdir=None, mode='a')

    A compressed and enlargeable data container either in-memory or on-disk.

    `carray` exposes a series of methods for dealing with the compressed
    container in a NumPy-like way.

    Parameters
    ----------
    array : a NumPy-like object
        This is taken as the input to create the carray.  It can be any Python
        object that can be converted into a NumPy object.  The data type of
        the resulting carray will be the same as this NumPy object.
    cparams : instance of the `cparams` class, optional
        Parameters to the internal Blosc compressor.
    dtype : NumPy dtype
        Force this `dtype` for the carray (rather than the `array` one).
    dflt : Python or NumPy scalar
        The value to be used when enlarging the carray.  If None,
        the default is
        filling with zeros.
    expectedlen : int, optional
        A guess on the expected length of this object.  This will serve to
        decide the best `chunklen` used for compression and memory I/O
        purposes.
    chunklen : int, optional
        The number of items that fits into a chunk.  By specifying it you can
        explicitely set the chunk size used for compression and memory I/O.
        Only use it if you know what are you doing.
    rootdir : str, optional
        The directory where all the data and metadata will be stored.  If
        specified, then the carray object will be disk-based (i.e. all chunks
        will live on-disk, not in memory) and persistent (i.e. it can be
        restored in other session, e.g. via the `open()` top-level function).
    safe : bool (defaults to True)
        Coerces inputs to array types.  Set to false if you always give
        correctly typed, strided, and shaped arrays and if you never use Object
        dtype.
    mode : str, optional
        The mode that a *persistent* carray should be created/opened.  The
        values can be:

        * 'r' for read-only
        * 'w' for read/write.  During carray creation, the `rootdir` will be
          removed if it exists.  During carray opening, the carray will be
          resized to 0.
        * 'a' for append (possible data inside `rootdir` will not be
          removed).

    """
    property nchunks:
        """Number of chunks in the carray"""
        def __get__(self):
            # TODO: do we need to handle the last chunk specially?
            return <npy_intp> cython.cdiv(self._nbytes, self._chunksize)

    property partitions:
        """List of tuples indicating the bounds for each chunk"""
        def __get__(self):
            nchunks = <npy_intp> cython.cdiv(self._nbytes, self._chunksize)
            chunklen = cython.cdiv(self._chunksize, self.atomsize)
            return [(i * chunklen, (i + 1) * chunklen) for i in
                    xrange(nchunks)]

    property attrs:
        """The attribute accessor.

        See Also
        --------

        attrs.attrs

        """
        def __get__(self):
            return self._attrs

    property cbytes:
        "The compressed size of this object (in bytes)."
        def __get__(self):
            return self._cbytes

    property chunklen:
        "The chunklen of this object (in rows)."
        def __get__(self):
            return self._chunklen

    property cparams:
        "The compression parameters for this object."
        def __get__(self):
            return self._cparams

    property dflt:
        "The default value of this object."
        def __get__(self):
            return self._dflt

    property dtype:
        "The dtype of this object."
        def __get__(self):
            return self._dtype.base

    property len:
        "The length (leading dimension) of this object."
        def __get__(self):
            if self._dtype.char == 'O':
                return len(self.chunks)
            else:
                # Important to do the cast in order to get a npy_intp result
                return <npy_intp> cython.cdiv(self._nbytes, self.atomsize)

    property mode:
        "The mode used to create/open the `mode`."
        def __get__(self):
            return self._mode
        def __set__(self, value):
            self._mode = value
            if hasattr(self.chunks, 'mode'):
                self.chunks.mode = value

    property nbytes:
        "The original (uncompressed) size of this object (in bytes)."
        def __get__(self):
            return self._nbytes

    property ndim:
        "The number of dimensions of this object."
        def __get__(self):
            return len(self.shape)

    property safe:
        "Whether or not to perform type/shape checks on every operation."
        def __get__(self):
            return self._safe

    property shape:
        "The shape of this object."
        def __get__(self):
            # note: the int cast is in order to get a consistent type
            # across windows and linux
            return tuple((int(self.len),) + self._dtype.shape)

    property size:
        "The size of this object."
        def __get__(self):
            return np.prod(self.shape)

    property rootdir:
        "The on-disk directory used for persistency."
        def __get__(self):
            return self._rootdir
        def __set__(self, value):
            if not self.rootdir:
                raise ValueError(
                    "cannot modify the rootdir value of an in-memory carray")
            self._rootdir = value
            self.chunks.rootdir = value

    def __cinit__(self, object array=None, object cparams=None,
                  object dtype=None, object dflt=None,
                  object expectedlen=None, object chunklen=None,
                  object rootdir=None, object safe=True, object mode="a",
                  object mmap=False):

        self._rootdir = rootdir
        if mode not in ('r', 'w', 'a'):
            raise ValueError("mode should be 'r', 'w' or 'a'")
        self._mode = mode
        self._safe = safe
        self._mmap = mmap

        if array is not None:
            self._create_carray(array, cparams, dtype, dflt,
                                expectedlen, chunklen, rootdir, mode)
            _new = True
        elif rootdir is not None:
            meta_info = self._read_meta()
            self._open_carray(*meta_info)
            _new = False
        else:
            raise ValueError(
                "You need at least to pass an array or/and a rootdir")

        # Attach the attrs to this object
        self._attrs = attrs.attrs(self._rootdir, self.mode, _new=_new)

        # Cache a len-1 array for accelerating self[int] case
        self.arr1 = np.empty(shape=(1,), dtype=self._dtype)

        # Sentinels
        self.sss_mode = False
        self.wheretrue_mode = False
        self.where_mode = False
        self.idxcache = -1  # cache not initialized

    cdef _adapt_dtype(self, dtype_, shape):
        """adapt the dtype to one supported in carray.
        returns the adapted type with the shape modified accordingly.
        """
        if dtype_.hasobject:
            if dtype_ != np.object_:
                raise TypeError(repr(dtype_) + " is not a supported dtype")
        else:
            dtype_ = np.dtype((dtype_, shape[1:]))

        return dtype_

    def _open_carray(self, shape, cparams, dtype, dflt,
                    expectedlen, cbytes, chunklen, xchunks=None):
        """Open an existing array."""
        cdef object array_, _dflt
        cdef npy_intp calen

        if len(shape) == 1:
            self._dtype = dtype
        else:
            # Multidimensional array.  The atom will have array_.shape[1:]
            # dims.
            # atom dimensions will be stored in `self._dtype`, which is
            # different
            # than `self.dtype` in that `self._dtype` dimensions are borrowed
            # from `self.shape`.  `self.dtype` will always be scalar (NumPy
            # convention).
            self._dtype = dtype = np.dtype((dtype.base, shape[1:]))

        self._cparams = cparams
        self.atomsize = dtype.itemsize
        self.itemsize = dtype.base.itemsize
        self._chunklen = chunklen
        self._chunksize = chunklen * self.atomsize
        self._dflt = dflt
        self.expectedlen = expectedlen

        if xchunks is None:
            # No chunks container, so read meta info from disk
            # Check rootdir hierarchy
            if not os.path.isdir(self._rootdir):
                raise IOError("root directory does not exist")
            self.datadir = os.path.join(self._rootdir, DATA_DIR)
            if not os.path.isdir(self.datadir):
                raise IOError("data directory does not exist")
            self.metadir = os.path.join(self._rootdir, META_DIR)
            if not os.path.isdir(self.metadir):
                raise IOError("meta directory does not exist")

            # Finally, open data directory
            metainfo = (dtype, cparams, shape[0], self._mode, chunklen)
            self.chunks = chunks(self._rootdir, metainfo=metainfo, _new=False, mmap=self._mmap)
        else:
            raise NotImplementedError('xchunks not supported now')

        # Update some counters
        calen = shape[0]  # the length ot the carray
        self.leftover = cython.cmod(calen, chunklen) * self.atomsize
        self._cbytes = cbytes
        self._nbytes = calen * self.atomsize

        if self._mode == "w":
            # Remove all entries when mode is 'w'
            self.resize(0)

    def _read_meta(self):
        """Read persistent metadata."""

        # First read the size info
        metadir = os.path.join(self._rootdir, META_DIR)
        shapef = os.path.join(metadir, SIZES_FILE)
        with open(shapef, 'rb') as shapefh:
            sizes = json.loads(shapefh.read().decode('ascii'))
        shape = sizes['shape']
        if type(shape) == list:
            shape = tuple(shape)
        nbytes = sizes["nbytes"]
        cbytes = sizes["cbytes"]

        # Then the rest of metadata
        storagef = os.path.join(metadir, STORAGE_FILE)
        with open(storagef, 'rb') as storagefh:
            data = json.loads(storagefh.read().decode('ascii'))
        dtype_ = np.dtype(data["dtype"])
        chunklen = data["chunklen"]
        cparams = data["cparams"]
        cname = cparams['cname'] if 'cname' in cparams else 'blosclz'
        quantize = cparams['quantize'] if 'quantize' in cparams else None
        cparams = jqbcolz.cparams(
            clevel=data["cparams"]["clevel"],
            shuffle=data["cparams"]["shuffle"],
            cname=cname,
            quantize=quantize)
        expectedlen = data["expectedlen"]
        dflt = data["dflt"]
        return (shape, cparams, dtype_, dflt, expectedlen, cbytes, chunklen)

    def __len__(self):
        return self.len

    def __sizeof__(self):
        return self._cbytes

    cdef int getitem_cache(self, npy_intp pos, char *dest):
        """Get a single item and put it in `dest`.  It caches a complete block.

        It returns 1 if asked `pos` can be copied to `dest`.  Else,
        this returns
        0.

        NOTE: As Blosc supports decompressing just a block inside a chunk, the
        data that is cached is a *block*, as it is the least amount of data
        that
        can be decompressed.  This saves both time and memory.

        IMPORTANT: Any update operation (e.g. __setitem__) *must* disable this
        cache by setting self.idxcache = -2.
        """
        cdef int ret, atomsize, blocksize, offset, extent
        cdef int idxcache, posinbytes, blocklen
        cdef npy_intp nchunk, nchunks, chunklen
        cdef chunk chunk_

        atomsize = self.atomsize
        nchunks = <npy_intp> cython.cdiv(self._nbytes, self._chunksize)
        chunklen = self._chunklen
        nchunk = <npy_intp> cython.cdiv(pos, chunklen)
        pos -= nchunk * chunklen

        # Locate the *block* inside the chunk
        chunk_ = self.chunks[nchunk]
        blocksize = chunk_.blocksize
        blocklen = <npy_intp> cython.cdiv(blocksize, atomsize)

        if (atomsize > blocksize):
            # This request cannot be resolved here
            return 0

        # Check whether the cache block has to be initialized
        if self.idxcache < 0:
            self.blockcache = np.empty(shape=(blocklen,), dtype=self._dtype)
            self.datacache = self.blockcache.data

        # Check if block is cached
        offset = <npy_intp> cython.cdiv(pos, blocklen) * blocklen
        posinbytes = (pos % blocklen) * atomsize
        idxcache = nchunk * chunklen + offset
        if idxcache == self.idxcache:
            # Hit!
            memcpy(dest, self.datacache + posinbytes, atomsize)
            return 1

        # No luck. Read a complete block.
        # this_chunklen: length of chunk_
        this_chunklen = <npy_intp> cython.cdiv(chunk_.nbytes, atomsize)
        extent = blocklen
        if offset + blocklen > this_chunklen:
            extent = this_chunklen % blocklen
        chunk_._getitem(offset, offset + extent, self.datacache)
        # Copy the interesting bits to dest
        memcpy(dest, self.datacache + posinbytes, atomsize)
        # Update the cache index
        self.idxcache = idxcache
        return 1

    def free_cachemem(self):
        """Release in-memory cached chunk"""
        if type(self.chunks) is not list:
            self.chunks.free_cachemem()
        self.idxcache = -1
        self.blockcache = None

    def _getitem_object(self, start, stop=None, step=None):
        """Retrieve elements of type object."""
        import pickle

        if stop is None and step is None:
            # Integer
            cchunk = self.chunks[start]
            chunk = cchunk.getudata()
            return pickle.loads(chunk)

        # Range
        objs = [self._getitem_object(i) for i in xrange(start, stop, step)]
        return np.array(objs, dtype=self._dtype)

    def __getitem__(self, object key):
        """ x.__getitem__(key) <==> x[key]

        Returns values based on `key`.  All the functionality of
        ``ndarray.__getitem__()`` is supported (including fancy indexing),
        plus a
        special support for expressions:

        Parameters
        ----------
        key : string
            It will be interpret as a boolean expression (computed via
            `eval`) and
            the elements where these values are true will be returned as a
            NumPy
            array.

        See Also
        --------
        eval

        """

        cdef int chunklen
        cdef npy_intp startb, stopb
        cdef npy_intp nchunk, keychunk, nchunks, first_chunk, last_chunk
        cdef npy_intp nwrow, blen
        cdef ndarray arr1, dest
        cdef object start, stop, step
        cdef object arr
        cdef chunk _chunk

        chunklen = self._chunklen

        # Check for integer
        if isinstance(key, _inttypes):
            if key < 0:
                # To support negative values
                key += self.len
            if key >= self.len:
                raise IndexError("index out of range")
            arr1 = self.arr1
            if self.dtype.char == 'O':
                return self._getitem_object(key)
            if self.getitem_cache(key, arr1.data):
                if self.itemsize == self.atomsize:
                    return PyArray_GETITEM(arr1, arr1.data)
                else:
                    return arr1[0].copy()
            # Fallback action: use the slice code
            return np.squeeze(self[slice(key, key + 1, 1)])
        # Slices
        elif isinstance(key, slice):
            (start, stop, step) = key.start, key.stop, key.step
            if step and step <= 0:
                raise NotImplementedError("step in slice can only be positive")
        # Multidimensional keys
        elif isinstance(key, tuple):
            if len(key) == 0:
                raise ValueError("empty tuple not supported")
            elif len(key) == 1:
                return self[key[0]]
            # An n-dimensional slice
            # First, retrieve elements in the leading dimension
            arr = self[key[0]]
            # Then, keep only the required elements in other dimensions
            if type(key[0]) == slice:
                arr = arr[(slice(None),) + key[1:]]
            else:
                arr = arr[key[1:]]
            # Force a copy in case returned array is not contiguous
            if not arr.flags.contiguous:
                arr = arr.copy()
            return arr
        # List of integers (case of fancy indexing)
        elif isinstance(key, list):
            # Try to convert to a integer array
            try:
                key = np.array(key, dtype=np.int_)
            except:
                raise IndexError(
                    "key cannot be converted to an array of indices")
            return self[key]
        # A boolean or integer array (case of fancy indexing)
        elif hasattr(key, "dtype"):
            if key.dtype.type == np.bool_:
                # A boolean array
                if len(key) != self.len:
                    raise IndexError(
                        "boolean array length must match len(self)")
                if isinstance(key, carray):
                    count = key.sum()
                else:
                    count = -1
                return np.fromiter(self.where(key), dtype=self._dtype,
                                   count=count)
            elif np.issubsctype(key, np.int_):
                # An integer array
                return np.array([self[i] for i in key], dtype=self._dtype.base)
            else:
                raise IndexError(
                    "arrays used as indices must be integer (or boolean)")
        # A boolean expression (case of fancy indexing)
        elif type(key) is str:
            # Evaluate
            result = jqbcolz.eval(key)
            if result.dtype.type != np.bool_:
                raise IndexError("only boolean expressions supported")
            if len(result) != self.len:
                raise IndexError(
                    "boolean expression outcome must match len(self)")
            # Call __getitem__ again
            return self[result]
        # All the rest not implemented
        else:
            raise NotImplementedError("key not supported: %s" % repr(key))

        # From now on, will only deal with [start:stop:step] slices

        # Get the corrected values for start, stop, step
        (start, stop, step) = slice(start, stop, step).indices(self.len)

        # Build a numpy container
        blen = get_len_of_range(start, stop, step)
        arr = np.empty(shape=(blen,), dtype=self._dtype)
        if blen == 0:
            # If empty, return immediately
            return arr

        if self.dtype.char == 'O':
            return self._getitem_object(start, stop, step)

        # Fill it from data in chunks
        nwrow = 0
        nchunks = <npy_intp> cython.cdiv(self._nbytes, self._chunksize)
        if self.leftover > 0:
            nchunks += 1
        first_chunk = <npy_intp> cython.cdiv(start, self.chunklen)
        last_chunk = <npy_intp> cython.cdiv(stop, self.chunklen) + 1
        last_chunk = min(last_chunk, nchunks)
        for nchunk from first_chunk <= nchunk < last_chunk:
            # Compute start & stop for each block
            startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop,
                                             step)
            if blen == 0:
                continue
            if 1:
                if step > 1:
                    arr[nwrow:nwrow + blen] = self.chunks[nchunk][
                                              startb:stopb:step]
                else:
                    # no step, can store directly
                    dest = arr[nwrow:nwrow + blen]
                    _chunk = self.chunks[nchunk]
                    _chunk._getitem(startb, stopb, dest.data)
            nwrow += blen

        return arr

    # This is a private function that is specific for `eval`
    def _getrange(self, npy_intp start, npy_intp blen, ndarray out):
        cdef int chunklen
        cdef npy_intp startb, stopb
        cdef npy_intp nwrow, stop, cblen
        cdef npy_intp schunk, echunk, nchunk, nchunks
        cdef chunk chunk_

        # Check that we are inside limits
        nrows = <npy_intp> cython.cdiv(self._nbytes, self.atomsize)
        if (start + blen) > nrows:
            blen = nrows - start

        # Fill `out` from data in chunks
        nwrow = 0
        stop = start + blen
        nchunks = <npy_intp> cython.cdiv(self._nbytes, self._chunksize)
        chunklen = cython.cdiv(self._chunksize, self.atomsize)
        schunk = <npy_intp> cython.cdiv(start, chunklen)
        echunk = <npy_intp> cython.cdiv((start + blen), chunklen)
        for nchunk from schunk <= nchunk <= echunk:
            # Compute start & stop for each block
            startb = start % chunklen
            stopb = chunklen
            if (start + startb) + chunklen > stop:
                # XXX I still have to explain why this expression works
                # for chunklen > (start + blen)
                stopb = (stop - start) + startb
                # stopb can never be larger than chunklen
                if stopb > chunklen:
                    stopb = chunklen
            cblen = stopb - startb
            if cblen == 0:
                continue
            if 1:
                chunk_ = self.chunks[nchunk]
                chunk_._getitem(startb, stopb,
                                out.data + nwrow * self.atomsize)
            nwrow += cblen
            start += cblen


    cdef reset_iter_sentinels(self):
        """Reset sentinels for iterator."""
        self.sss_mode = False
        self.wheretrue_mode = False
        self.where_mode = False
        self.where_arr = None
        self.nhits = 0
        self.limit = _MAXINT
        self.skip = 0
        self.start = 0
        self.stop = <npy_intp> cython.cdiv(self._nbytes, self.atomsize)
        self.step = 1
        self.iter_exhausted = False

    cdef int check_zeros(self, object barr):
        """Check for zeros.  Return 1 if all zeros, else return 0."""
        cdef int bsize
        cdef npy_intp nchunk
        cdef carray carr
        cdef ndarray ndarr
        cdef chunk chunk_

        if isinstance(barr, carray):
            # Check for zero'ed chunks in carrays
            carr = barr
            nchunk = <npy_intp> cython.cdiv(self.nrowsread, self.nrowsinbuf)
            if nchunk < len(carr.chunks):
                chunk_ = carr.chunks[nchunk]
                if chunk_.isconstant and chunk_.constant in (0, ''):
                    return 1
        else:
            # Check for zero'ed chunks in ndarrays
            ndarr = barr
            bsize = self.nrowsinbuf
            if self.nrowsread + bsize > self.len:
                bsize = self.len - self.nrowsread
            if check_zeros(ndarr.data + self.nrowsread, bsize):
                return 1
        return 0

    def __str__(self):
        return array2string(self)

    def __repr__(self):
        snbytes = utils.human_readable_size(self._nbytes)
        scbytes = utils.human_readable_size(self._cbytes)
        if not self._cbytes:
            cratio = np.nan
        else:
            cratio = self._nbytes / float(self._cbytes)
        header = "carray(%s, %s)\n" % (self.shape, self.dtype)
        header += "  nbytes := %s; cbytes := %s; ratio: %.2f\n" % (
            snbytes, scbytes, cratio)
        header += "  cparams := %r\n" % self.cparams
        blocksize = self.chunks[0].blocksize if len(self.chunks) > 0 else 0
        header += "  chunklen := %s; chunksize: %s; blocksize: %s\n" % (
            self.chunklen, self._chunksize, blocksize)
        if self._rootdir:
            header += "  rootdir := '%s'\n" % self._rootdir
            header += "  mode    := '%s'\n" % self.mode
        fullrepr = header + str(self)
        return fullrepr

    def __reduce__(self):
        if self.rootdir:
            return (build_carray, (None,self.rootdir,))
        else:
            return (build_carray,(self[:],None,))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self.mode != 'r':
            self.flush()

    def __array__(self, dtype=None, **kwargs):
        x = self[:]
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        return x


## Local Variables:
## mode: python
## tab-width: 4
## fill-column: 78
## End:
