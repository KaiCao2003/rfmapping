"""A bounded runtime thread setting must retain the accelerated shuffle path."""

import os
import subprocess
import sys

import pytest


def test_numba_thread_setting_and_numpy_shuffle_agree():
    pytest.importorskip("numba")
    code = """
import numpy as np
from numba import get_num_threads
from Utils import kilosort_utils as k
assert get_num_threads() == 4
assert k._compute_shuffle_r_numba is not None
edges = np.arange(0., 361., 12.)
angles = np.deg2rad((edges[:-1] + edges[1:]) / 2)
args = (
    np.array([.1, .9, 2.4, 3.2, 4.8]), np.array([0., .5, 1., 4.9]), 5.,
    np.array([0., 5.]), np.array([0., 3., 5.]),
    np.array([0., .5, 1., 2., 2.5, 5., 5.5, 6.5]),
    np.array([0., 30., 60., 90., 120., 180., 210., 359.]),
    np.array([0, 5]), np.array([5, 8]), edges,
    np.linspace(1., 3., 30), np.cos(angles), np.sin(angles),
)
np.testing.assert_allclose(k._compute_shuffle_r_numba(*args),
                           k._compute_shuffle_r_numpy(*args), rtol=1e-12, atol=1e-12)
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "NUMBA_NUM_THREADS": "4"})
