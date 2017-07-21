# coding: utf-8

""" Test jqbcolz

Usage:
py.test test_jqbcolz.py

Requirement:
run 'python setup.py build_ext --inplace' before
"""


from __future__ import print_function, unicode_literals, absolute_import

import jqbcolz
import bcolz
import random
import numpy as np


def test():
    for field in ('volume', 'date', 'open'):
        for _mmap in (False, True):
            for path in (
                    '/opt/data/jq/bundle/daydata/00/000300.XSHG/' + field,
                    '/opt/data/jq/bundle/minutedata/00/000300.XSHG/' + field,
            ):
                ba = bcolz.open(path)
                ja = jqbcolz.open(path, mmap=_mmap)

                assert (ba[:] == ja[:]).all()

                n = len(ba)

                for i in (0, 1, n-1, -1, -2):
                    assert ba[i] == ja[i]

                assert (ba[:10] == ja[:10]).all()
                assert (ba[-10:] == ja[-10:]).all()

                for _ in range(100):
                    i = random.randint(-n, n)
                    assert ba[i] == ja[i]

                for _ in range(100):
                    i = random.randint(0, n)
                    j = random.randint(i, n)
                    s = random.randint(1, n)
                    assert (ba[i:j:s] == ja[i:j:s]).all()
    pass


def test_all():
    for path in (
        '/opt/data/jq/bundle/daydata/00/000300.XSHG/volume',
        '/opt/data/jq/bundle/minutedata/00/000300.XSHG/volume',
    ):
        ba = bcolz.open(path)
        ja = jqbcolz.open(path, mmap=True)
        assert len(ba) == len(ja)
        n = len(ba)
        for i in range(0, n):
            assert ba[i] == ja[i]

        for i in range(0, n):
            assert (ba[i:10] == ja[i:10]).all()
    pass


def test_ctable():
    for path in (
        '/opt/data/jq/bundle/daydata/00/000300.XSHG',
        '/opt/data/jq/bundle/minutedata/00/000300.XSHG',
    ):
        ba = bcolz.open(path)
        ja = jqbcolz.open(path, mmap=True)
        for field in ba.names:
            assert np.allclose(ba[field][:], ja[field][:], equal_nan=True)
            n = len(ba)
            for _ in range(100):
                i = random.randint(0, n)
                j = random.randint(i, n)
                s = random.randint(1, n)
                assert (ba[field][i:j:s] == ja[field][i:j:s]).all()
    pass
