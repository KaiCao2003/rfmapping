"""Native saved-TC decoding without pseudo-counts, smoothing, or gap filling."""

import json

import numpy as np
import pandas as pd
import pytest

from Utils.hd_rf_video_decode import decode_hd_frames, load_decoder_tuning_curves


def test_native_saved_tc_selection_and_rates_preserve_source(tmp_path):
    path = tmp_path / "tuning_curves.tc"
    centers = np.arange(1., 360., 2.)
    rates = 5 + 4 * np.cos(np.deg2rad(centers - 45))
    occupancy = np.linspace(1., 3., 180)
    source = dict(
        metadata={"classification": {}, "session": "session_c", "probe": "A"},
        angle_bin_edges_deg=np.arange(0., 361., 2.).tolist(),
        occupancy_time_s=occupancy.tolist(), unit_id=[471, 3],
        spike_counts=[(rates * occupancy).tolist(), occupancy.tolist()],
        firing_rate_hz=[rates.tolist(), np.ones(180).tolist()],
        unit_data=dict(hd_class=[2, 0], rayleigh_p=[.001, .9], shuffle_p=[.001, .9]),
    )
    path.write_text(json.dumps(source))
    before = path.read_bytes()

    curves = load_decoder_tuning_curves(path)

    assert curves.index.tolist() == [471]
    np.testing.assert_array_equal(curves.columns, centers)
    np.testing.assert_allclose(curves.iloc[0], rates, rtol=0, atol=1e-15)
    assert curves.attrs["metadata"]["session"] == "session_c"
    assert path.read_bytes() == before


def test_zero_rate_observation_stays_impossible_without_pseudocount():
    curves = pd.DataFrame([[10., 0.], [0., 10.]], index=[471, 3], columns=[45., 225.])

    decoded, qc = decode_hd_frames(
        curves, [.025, .225, .425, .425], [471, 3, 471, 3],
        [0., .1, .2, .3, .4, .5], .2, [[0., .6]],
    )

    np.testing.assert_array_equal(decoded, [45., 45., 225., 225., np.nan, np.nan])
    assert qc == dict(decoded_bins=3, valid_bins=2, silent_bins=0, impossible_bins=1)


def test_silent_population_camera_gap_and_partial_bins_remain_missing():
    curves = pd.DataFrame([[10., 1.]], index=[47], columns=[1., 181.])
    frames = np.array([-.01, 0., .1, .199999999, .2, .249, .25, .5,
                       1., 1.099999999, 1.1, 1.199999999, 1.2, 1.299999999, 1.3, 1.35])

    decoded, qc = decode_hd_frames(
        curves, [.025, .175, 1.025, 1.225], [47] * 4,
        frames, .1, [[0., .25], [1., 1.35]],
    )

    np.testing.assert_array_equal(np.isfinite(decoded), [
        False, True, True, True, False, False, False, False,
        True, True, False, False, True, True, False, False,
    ])
    assert qc == dict(decoded_bins=5, valid_bins=4, silent_bins=1, impossible_bins=0)


def test_chunking_and_unsorted_frame_order_keep_exact_bin_membership():
    curves = pd.DataFrame([[10., 1.], [1., 10.]], index=[47, 3], columns=[45., 225.])
    frames = np.array([.6, .599999999, 0., .7, .699999999])
    times = [.625, .525]
    clusters = [47, 3]

    whole, qc = decode_hd_frames(curves, times, clusters, frames, .1, [[0., 1.]])
    chunked, chunk_qc = decode_hd_frames(
        curves, times, clusters, frames, .1, [[0., 1.]], chunk_bins=1,
    )

    np.testing.assert_array_equal(whole, [45., 225., np.nan, np.nan, 45.])
    np.testing.assert_array_equal(chunked, whole)
    assert chunk_qc == qc


def test_silent_unit_still_contributes_poisson_rate_penalty():
    curves = pd.DataFrame([[2., 2.], [10., 0.]], index=[47, 3], columns=[45., 225.])

    decoded, _ = decode_hd_frames(curves, [.025], [47], [.05], .1, [[0., .1]])

    np.testing.assert_array_equal(decoded, [225.])


def test_unmeasured_native_angle_bins_are_rejected(tmp_path):
    path = tmp_path / "tuning_curves.tc"
    occupancy = np.ones(180)
    occupancy[4] = 0
    source = dict(
        metadata={"classification": {}}, angle_bin_edges_deg=np.arange(0., 361., 2.).tolist(),
        occupancy_time_s=occupancy.tolist(), unit_id=[47],
        spike_counts=[occupancy.tolist()], firing_rate_hz=[np.ones(180).tolist()],
        unit_data=dict(hd_class=[2], rayleigh_p=[.001], shuffle_p=[.001]),
    )
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="positive saved occupancy"):
        load_decoder_tuning_curves(path)
