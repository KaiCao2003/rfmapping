from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from Utils.rf_rates import aggregate_rate, counts_to_rates, resolve_presentation_counts


def _recording(tmp_path, trials, edges, stem):
    session = tmp_path / "261001_13"
    (session / "data").mkdir(parents=True)
    savemat(session / "261001.mat", {"trials": trials})
    np.save(session / "data/on_list_times.npy", edges)
    return session / "data/rfmapping/good/ProbeA" / f"{stem}.rfmap"


def test_bin_and_window_hz_use_presentation_exposure_and_actual_widths():
    counts = np.array([[[[2, 1, 6], [7, 9, 1]]]], dtype=float)
    presentations = np.array([[2, 0]])
    edges = np.array([0.0, 0.1, 0.15, 0.3])
    rates = counts_to_rates(counts, presentations, edges)
    np.testing.assert_allclose(rates, [[[[10, 10, 20], [0, 0, 0]]]])
    np.testing.assert_allclose(aggregate_rate(rates, edges, 0, 3), [[[[15], [0]]]])
    np.testing.assert_allclose(aggregate_rate(rates, edges, 1, 3), [[[[17.5], [0]]]])
    np.testing.assert_array_equal(aggregate_rate(rates, edges, 1, 1), np.zeros((1, 1, 2, 1)))


def test_saved_counts_do_not_require_raw_session_files():
    counts, provenance = resolve_presentation_counts(
        {"stimulusPresentationCounts": [[3, 0]]}, Path("standalone.rfmap"), [0, 1], [0],
    )
    np.testing.assert_array_equal(counts, [[3, 0]])
    assert provenance == {"method": "stimulusPresentationCounts"}


def test_free_moving_legacy_geometry_requires_saved_counts():
    source = Path("/recordings/261001_13/data/regular_unitsSpikeCounts_261001_13_free_moving.rfmap")
    with pytest.raises(ValueError, match="free-moving.*stimulusPresentationCounts"):
        resolve_presentation_counts({}, source, [0], [0])
    counts, _ = resolve_presentation_counts({"stimulusPresentationCounts": [[3]]}, source, [0], [0])
    np.testing.assert_array_equal(counts, [[3]])


@pytest.mark.parametrize("values", ([[1, -1]], [[1, 0.5]], [[1, np.nan]], [[1], [2]]))
def test_saved_presentation_counts_reject_invalid_values_or_transposed_shape(values):
    with pytest.raises(ValueError, match="stimulusPresentationCounts"):
        resolve_presentation_counts({"stimulusPresentationCounts": values}, "standalone.rfmap", [0, 1], [0])


@pytest.mark.parametrize("suffix, exposure", (
    ("", [[0.1, 0.4]]), ("_off", [[0.2, 0.5]]), ("_gray", [[0.3, 0.6]]),
))
def test_legacy_regular_counts_follow_selected_luminance_not_display_duration(tmp_path, suffix, exposure):
    trials = [
        {"Square_PositionX": x, "Square_PositionY": 0, "Square_Luminance": lum}
        for x in (-6, 6) for lum in (1, 0, 0.5)
    ]
    source = _recording(tmp_path, trials, [0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1],
                        f"regular_unitsSpikeCounts_261001_13{suffix}")
    counts, provenance = resolve_presentation_counts({"occupancyTimeSec": exposure}, source, [-6, 6], [0])
    np.testing.assert_array_equal(counts, [[1, 1]])
    assert provenance["analyzed_trials"] == 6
    assert provenance["trials_mat"] == str(tmp_path / "261001_13/261001.mat")


def test_legacy_trial_reconstruction_checks_the_saved_exposure(tmp_path):
    trials = [{"Square_PositionX": 0, "Square_PositionY": 0, "Square_Luminance": 1}]
    source = _recording(tmp_path, trials, [0, 0.1], "regular_unitsSpikeCounts_261001_13")
    with pytest.raises(ValueError, match="does not match"):
        resolve_presentation_counts({"occupancyTimeSec": [[0.2]]}, source, [0], [0])


def test_legacy_missing_terminal_edge_counts_only_trials_actually_pooled(tmp_path):
    trials = [{"Square_PositionX": 0, "Square_PositionY": 0, "Square_Luminance": 1}] * 3
    source = _recording(tmp_path, trials, [0, 0.1, 0.3], "regular_unitsSpikeCounts_261001_13")
    counts, _ = resolve_presentation_counts({"occupancyTimeSec": [[0.3]]}, source, [0], [0])
    np.testing.assert_array_equal(counts, [[2]])


def test_vertical_bars_count_each_trial_once_per_coarse_bin(tmp_path):
    trials = [
        {"Square_PositionX": -174, "Square_PositionY": 0,
         "Square_Luminance": 1, "Square_Size": width}
        for width in (3, 12)
    ]
    source = _recording(tmp_path, trials, [0, 0.1, 0.3],
                        "regular_unitsSpikeCounts_261001_13_vertical_bar_pooled_bin3deg")
    exposure = np.zeros((1, 120))
    exposure[0, :4] = [0.2, 0.3, 0.3, 0.2]
    counts, _ = resolve_presentation_counts(
        {"occupancyTimeSec": exposure, "isVerticalBar": True,
         "screenWidthPix": 960, "screenDeg": 360}, source, -178.5 + np.arange(120) * 3, [0],
    )
    expected = np.zeros((1, 120))
    expected[0, :4] = [1, 2, 2, 1]
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize("mode", ("egocentric", "rotation", "allocentric_pixelbins"))
def test_transformed_counts_preserve_wrap_and_binary_coarsening(tmp_path, mode):
    trials = [
        {"Square_PositionX": x, "Square_PositionY": 0, "Square_Luminance": 1,
         "Square_Size": 2, "BackgroundRotation_XOffset_Pix": offset}
        for x, offset in zip((-3, 3, -4) if mode == "allocentric_pixelbins" else (-4, -4, -4), (1, 7, 0))
    ]
    source = _recording(tmp_path, trials, [0, 0.1, 0.3, 0.6], f"{mode}_30_unitsSpikeCounts_261001_13")
    np.save(tmp_path / "261001_13/data/hd_trials_times.npy", [1, 7, 0])
    exposure = [[0.4, 0, 0, 0.2 if mode == "allocentric_pixelbins" else 0.5]]
    counts, _ = resolve_presentation_counts(
        {"occupancyTimeSec": exposure, "screenWidthPix": 8, "screenDeg": 8,
         "total_deg": 8, "rotationOffsetSign": 1}, source, [-3, -1, 1, 3], [0],
    )
    np.testing.assert_array_equal(counts, [[2, 0, 0, 1 if mode == "allocentric_pixelbins" else 2]])
