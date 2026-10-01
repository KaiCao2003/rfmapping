import numpy as np
import pandas as pd
import json

from Utils.pitch_direction_comparison import circular_linear_association, preferred_angles
from Utils.pitch_direction_comparison import rf_vertical_profiles, pitch_rf_regression, load_pitch_rf_pair, cache_name


def test_pitch_peaks_remain_signed_and_missing_curves_have_no_peak():
    hd = pd.DataFrame([[0, 2, 1], [np.nan, np.nan, np.nan]], columns=[6., 18., 30.])
    pitch = pd.DataFrame([[4, 1, 0], [0, 0, 0]], columns=[-60., 0., 60.])
    peaks = preferred_angles(hd, pitch)
    np.testing.assert_allclose(peaks.iloc[0], [18, -60])
    assert peaks.iloc[1].isna().all()


def test_circular_linear_association_is_invariant_to_heading_zero():
    hd = np.arange(0., 360., 12.)
    pitch = 20 * np.sin(np.deg2rad(hd)) - 15
    original = circular_linear_association(hd, pitch, n_permutations=100, seed=3)
    rotated = circular_linear_association((hd + 80) % 360, pitch, n_permutations=100, seed=3)
    np.testing.assert_allclose(original["r"], 1.)
    assert original == rotated
    assert original["p"] == 1 / 101


def test_rf_vertical_profile_uses_rates_and_differs_from_2d_max(tmp_path):
    # First elevation has the largest single bin (10), second the largest sum
    # (6+6). Unequal exposure makes the distinction depend on normalization.
    counts = np.array([[[20, 0], [6, 6]]])[..., None]
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": [-10, 10], "yPositions": [-20, 20],
        "timeBinEdges": [0, .2], "occupancyTimeSec": [[2, 1], [1, 1]],
    }))
    profiles, peaks = rf_vertical_profiles(path, mouse="m", date="d", probe="A", rf_type=None)
    np.testing.assert_array_equal(profiles.columns, [-20, 20])
    np.testing.assert_allclose(profiles.iloc[0], [12, 10])
    assert peaks.iloc[0].rf_vertical_profile_peak_deg == -20
    assert peaks.iloc[0].rf_2d_max_elevation_deg == 20
    np.savez(tmp_path / cache_name("m", "d", 1, "A"), unit_id=[7],
             hd_counts=[[1., 1.]], hd_occupancy_s=[1., 1.], hd_edges=[0, 180, 360],
             pitch_counts=[[2., 1.]], pitch_occupancy_s=[1., 1.], pitch_edges=[-90, 0, 90])
    first = load_pitch_rf_pair(tmp_path, path, mouse="m", date="d", pitch_session=1, probe="A", rf_type=None)
    second = load_pitch_rf_pair(tmp_path, path, mouse="m", date="d", pitch_session=1, probe="A",
                                rf_type=None, rf_peak_method="max_bin_2d")
    assert first[2].index.equals(second[2].index)
    assert first[2].rf_peak_deg.iloc[0] == -20
    assert second[2].rf_peak_deg.iloc[0] == 20


def test_pitch_rf_regression_preserves_signed_linear_angles():
    peaks = pd.DataFrame({"pitch_peak_deg": [-40, -20, 0, 20], "rf_peak_deg": [-25, -15, -5, 5]})
    result = pitch_rf_regression(peaks)
    np.testing.assert_allclose([result["slope"], result["intercept"], result["r_squared"]], [.5, -5, 1])
