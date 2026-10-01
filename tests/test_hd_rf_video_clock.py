import numpy as np
import pytest

from Utils.hd_rf_video_data import build_video_clock


@pytest.mark.parametrize("phase", [0, 1])
def test_regular_two_to_one_clock_uses_explicit_phase(phase):
    motive = np.arange(12) / 120.
    basler = motive + .00005
    exposures, rows, qc = build_video_clock(motive, basler, 6, stride=2, phase=phase)

    expected_rows = phase + 2 * np.arange(6)
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_allclose(exposures, basler[expected_rows])
    assert qc["frame_period_s"] == pytest.approx(1 / 60)
    assert qc["stride"] == 2 and qc["phase"] == phase
    assert qc["paired_motive_pulse_count"] == 12
    assert qc["uncertain_video_frame_ranges_inclusive"] == []


def test_regular_clock_rejects_unexplained_video_frame_loss():
    motive = np.arange(12) / 120.
    with pytest.raises(ValueError, match="gives 6 video frames, but the AVI has 5"):
        build_video_clock(motive, motive + .00005, 5)


def test_explicit_dropped_frame_segments_preserve_unavailable_ranges():
    motive = np.arange(169732) / 120.
    basler = motive + .00005
    segments = [(0, 351, 0), (380, 72982, 56), (72996, 84813, 104)]
    exposures, rows, qc = build_video_clock(motive, basler, 84814, segments=segments)

    for first, last, offset in segments:
        frames = np.arange(first, last + 1)
        np.testing.assert_array_equal(rows[frames], 2 * frames + offset)
        np.testing.assert_allclose(exposures[frames], basler[(2 * frames + offset)])
    assert np.isnan(exposures[352:380]).all() and np.isnan(rows[352:380]).all()
    assert np.isnan(exposures[72983:72996]).all() and np.isnan(rows[72983:72996]).all()
    assert qc["uncertain_video_frame_ranges_inclusive"] == [[352, 379], [72983, 72995]]
    assert qc["uncertain_video_frame_count"] == 41
    assert qc["finite_video_frame_count"] == 84773
    assert qc["frame_period_s"] == pytest.approx(1 / 60)


def test_unpaired_basler_glitch_does_not_shift_motive_or_video_rows():
    motive = np.arange(12) / 100.
    basler = np.sort(np.r_[motive + .0001, .045])
    exposures, rows, qc = build_video_clock(motive, basler, 6)

    np.testing.assert_array_equal(rows, np.arange(0, 12, 2))
    np.testing.assert_allclose(exposures, motive[::2] + .0001)
    assert qc["unpaired_basler_pulse_count"] == 1
    assert qc["unmatched_motive_pulse_count"] == 0


def test_missing_basler_pulse_leaves_nan_without_shifting_following_frames():
    motive = np.arange(12) / 100.
    basler = np.delete(motive + .0001, 2)
    exposures, rows, qc = build_video_clock(motive, basler, 6)

    assert np.isnan(exposures[1]) and np.isnan(rows[1])
    np.testing.assert_array_equal(rows[[0, 2, 3, 4, 5]], [0, 4, 6, 8, 10])
    np.testing.assert_allclose(exposures[[0, 2, 3, 4, 5]], motive[[0, 4, 6, 8, 10]] + .0001)
    assert qc["unmatched_motive_pulse_count"] == 1
    assert qc["uncertain_video_frame_ranges_inclusive"] == [[1, 1]]


def test_reused_basler_pulse_invalidates_both_competing_motive_rows():
    motive = np.array([0., .25, 1., 2., 3., 4.])
    basler = np.array([.125, 1., 2., 3., 4.])
    exposures, rows, qc = build_video_clock(motive, basler, 6, stride=1)

    assert np.isnan(exposures[:2]).all() and np.isnan(rows[:2]).all()
    np.testing.assert_array_equal(exposures[2:], [1., 2., 3., 4.])
    assert qc["ambiguous_motive_pair_count"] == 2
    assert qc["uncertain_video_frame_ranges_inclusive"] == [[0, 1]]


@pytest.mark.parametrize("segments, message", [
    ([(0, 5, 2)], "Mapped Motive rows"),
    ([(0, 3, 0), (3, 5, 0)], "must not overlap"),
])
def test_mapping_segments_reject_invalid_rows_or_overlap(segments, message):
    motive = np.arange(12) / 100.
    with pytest.raises(ValueError, match=message):
        build_video_clock(motive, motive + .0001, 6, segments=segments)


def test_pulse_timestamp_duplicates_are_not_a_valid_measured_clock():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_video_clock([0., .01, .02, .03], [0., .01, .01, .03], 2)
