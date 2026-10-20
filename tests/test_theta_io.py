"""Saved theta output contains everything needed for later plotting."""

import json

import numpy as np
import pandas as pd
import pytest

from Utils.theta_analysis import analyze_theta_units, extract_theta_phase, summarize_lfp_reference
from Utils.theta_io import load_theta_analysis, save_theta_analysis


@pytest.fixture
def theta_output():
    times = np.arange(0., 8., .002)
    lfp = dict(times_s=times, voltage_uv=np.cos(2 * np.pi * 8 * times), sample_rate_hz=500.)
    theta = extract_theta_phase(times, lfp["voltage_uv"], sample_rate_hz=500.)
    intervals = [[0., 8.]]
    result = analyze_theta_units(
        {0: np.arange(1.03125, 6.5, .125), 9: np.empty(0)}, theta,
        tracking_intervals_s=intervals, moving_intervals_s=intervals,
        min_spikes_exclusive=10,
    )
    reference = summarize_lfp_reference(lfp, theta, np.asarray(intervals))
    return result, reference


def assert_saved_values(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_saved_values(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_saved_values(left, right)
    elif isinstance(expected, (float, np.floating)) and np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


def test_theta_round_trip_preserves_plot_evidence_and_nullable_labels(tmp_path, theta_output):
    result, reference = theta_output
    provenance = dict(session_id="session_1", input_files=[dict(path="saved_lfp.npz", sha256="abc")])
    directory = save_theta_analysis(
        tmp_path / "theta", result, provenance=provenance, lfp_reference=reference,
    )
    saved = load_theta_analysis(directory)
    pd.testing.assert_frame_equal(saved["summary"], result["summary"])
    assert_saved_values(saved["units"], result["units"])
    assert_saved_values(saved["metadata"], result["metadata"])
    assert_saved_values(saved["lfp_reference"], reference)
    assert saved["provenance"] == provenance
    assert pd.isna(saved["summary"].loc[9, "is_phase_locked"])
    assert saved["summary"].loc[0, "is_phase_locked"]
    assert set(path.name for path in directory.iterdir()) == {
        "theta_cells.csv", "theta_arrays.npz", "manifest.json",
    }


def test_theta_save_refuses_overwrite_and_replaces_optional_reference(tmp_path, theta_output):
    result, reference = theta_output
    directory = tmp_path / "theta"
    save_theta_analysis(directory, result, provenance={}, lfp_reference=reference)
    with pytest.raises(FileExistsError):
        save_theta_analysis(directory, result, provenance={})
    save_theta_analysis(directory, result, provenance={"rerun": True}, overwrite=True)
    saved = load_theta_analysis(directory)
    assert saved["lfp_reference"] is None
    assert saved["provenance"] == {"rerun": True}


def test_theta_round_trip_preserves_string_unit_identifiers(tmp_path, theta_output):
    result, _ = theta_output
    result["summary"]["unit_id"] = ["001", "NA"]
    result["summary"] = result["summary"].set_index("unit_id", drop=False)
    result["units"] = {"001": result["units"][0]}
    save_theta_analysis(tmp_path, result, provenance={})
    saved = load_theta_analysis(tmp_path)
    pd.testing.assert_frame_equal(saved["summary"], result["summary"])
    assert set(saved["units"]) == {"001"}


def test_missing_theta_outputs_are_not_generated(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_theta_analysis(missing)
    assert not missing.exists()


def test_theta_loader_rejects_unknown_saved_format(tmp_path, theta_output):
    result, _ = theta_output
    save_theta_analysis(tmp_path, result, provenance={})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format"] = "unrecognized"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported theta analysis format"):
        load_theta_analysis(tmp_path)


@pytest.mark.parametrize("spikes", [{}, {0: np.empty(0), 9: np.empty(0)}])
def test_empty_or_all_unassessed_theta_round_trip(tmp_path, spikes):
    result = analyze_theta_units(
        spikes, dict(parameters={}, phase_qc={}),
        tracking_intervals_s=[], moving_intervals_s=[], spatial_coverage=0.,
    )
    save_theta_analysis(tmp_path, result, provenance={})
    saved = load_theta_analysis(tmp_path)
    pd.testing.assert_frame_equal(saved["summary"], result["summary"])
    assert saved["units"] == {}
    assert saved["summary"]["is_theta_rhythmic"].isna().all()
    assert saved["summary"]["is_phase_locked"].isna().all()
    assert saved["summary"]["precession_candidate"].isna().all()
