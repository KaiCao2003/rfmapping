import numpy as np
import pytest

from Utils.ttl_utils import pre_process_signals


@pytest.mark.parametrize("mode, expected", [
    ("rising_edge", [1, 3]),
    ("falling_edge", [2, 4]),
    ("signal_pairs", [[1, 2], [3, 4]]),
])
def test_ttl_modes_preserve_the_requested_edges(mode, expected):
    pairs = np.array([[1, 2], [3, 4]])
    options = {"return_rising_edge": False, f"return_{mode}": True}
    result = pre_process_signals(pairs, **options)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(pairs, [[1, 2], [3, 4]])


@pytest.mark.parametrize("mode", ["rising_edge", "falling_edge", "signal_pairs"])
def test_empty_ttl_pairs_keep_the_requested_shape(mode):
    options = {"return_rising_edge": False, f"return_{mode}": True}
    result = pre_process_signals(np.empty((0, 2), dtype=int), **options)
    assert result.shape == ((0, 2) if mode == "signal_pairs" else (0,))
