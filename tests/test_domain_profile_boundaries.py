"""Domain readers do no analysis; profile calculators use only supplied data."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from EBB.profiles import ebc_profiles, prepare_ebc
from HD import profiles as hd
from HD import tuning_curve_utils
from RF.rflocate import RFMapList, asrfmap


def no_io(*args, **kwargs):
    pytest.fail("Profile calculation must not read or write files")


def no_classification(*args, **kwargs):
    pytest.fail("Reading or preparing profiles must retain saved classification")


def test_hd_reader_returns_saved_json_without_analysis_or_rewriting(tmp_path, monkeypatch):
    saved = {"unit_id": [7], "unit_data": {"hd_class": [3]},
             "metadata": {"source_note": "preserve exactly"}}
    path = tmp_path / "saved.tc"
    path.write_text(json.dumps(saved))
    original = path.read_bytes()
    monkeypatch.setattr(tuning_curve_utils, "classify_hd", no_classification)

    assert hd.load_hd_tc(path) == saved
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_hd_comparison_reader_retains_existing_values_without_calculation(tmp_path, monkeypatch):
    path = tmp_path / "m22/261008/261008_11/data/tc_comparison/hd_class3_ProbeA.csv"
    path.parent.mkdir(parents=True)
    path.write_text("unit_id,45,135,-135,-45\nm22:261008:A:7,1,3,2,4\n")
    original = path.read_bytes()
    monkeypatch.setattr(hd, "hd_profiles", no_classification)
    monkeypatch.setattr(hd, "rebin_profiles", no_classification)
    monkeypatch.setattr(hd, "normalize_tc", no_classification)

    table = hd.load_hd_comparison("m22", tmp_path, "A", "261008", 11)

    np.testing.assert_array_equal(table, [[1, 3, 2, 4]])
    np.testing.assert_array_equal(table.columns, [45, 135, -135, -45])
    assert table.index.tolist() == ["m22:261008:A:7"]
    assert table.attrs == {"label": "m22 261008 ProbeA HD", "range": [180, 270, 0, 90, 180],
                           "response_units": "Hz"}
    assert path.read_bytes() == original


def test_hd_comparison_reader_does_not_generate_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        hd.load_hd_comparison("m22", tmp_path, "A", "261008", 11)
    assert list(tmp_path.iterdir()) == []


def test_prep_hd_tc_only_transforms_the_supplied_table(monkeypatch):
    table = pd.DataFrame([[1., 3., 2., 4.], [0., 0., 0., 0.]],
                         index=["m22:261008:A:7", "m22:261008:A:9"],
                         columns=[45., 135., -135., -45.])
    table.attrs = {"response_units": "Hz", "label": "HD"}
    original = table.copy(deep=True)
    monkeypatch.setattr(Path, "open", no_io)
    monkeypatch.setattr(hd, "tc_loader", no_io)

    actual = hd.prep_hd_tc(table, bins=2)

    np.testing.assert_array_equal(actual, [[2 / 3, 1.], [0., 0.]])
    np.testing.assert_array_equal(actual.columns, [90., -90.])
    assert actual.attrs == {"response_units": "normalized", "label": "HD"}
    pd.testing.assert_frame_equal(table, original)


def test_hd_profiles_pool_counts_and_occupancy_without_io_or_classification(monkeypatch):
    data = {"unit_id": [9, 7], "unit_data": {"hd_class": [3, 0]},
            "angle_bin_edges_deg": np.arange(0, 361, 60).tolist(),
            "occupancy_time_s": [1, 9, 0, 0, 1, 3],
            "spike_counts": [[1, 18, 0, 0, 8, 12], [0, 0, 0, 0, 0, 0]]}
    original = deepcopy(data)
    monkeypatch.setattr(Path, "open", no_io)
    monkeypatch.setattr(hd, "read_formatted_json", no_io)
    monkeypatch.setattr(tuning_curve_utils, "classify_hd", no_classification)

    profiles = hd.hd_profiles(data, bins=3)

    np.testing.assert_array_equal(profiles, [[1.9, np.nan, 5], [0, np.nan, 0]])
    np.testing.assert_array_equal(profiles.columns, [60, -180, -60])
    assert profiles.index.tolist() == [("A", 9), ("A", 7)]
    assert profiles.attrs["unit_info"] == {("A", 9): {"hd_class": 3}, ("A", 7): {"hd_class": 0}}
    selected = hd.hd_pick(profiles, 3)
    assert selected.index.tolist() == [("A", 9)]
    selected.attrs["unit_info"][("A", 9)]["hd_class"] = 0
    assert profiles.attrs["unit_info"][("A", 9)]["hd_class"] == 3
    assert data == original


def test_hd_smoothing_preserves_constant_rate_with_unequal_occupancy(monkeypatch):
    occupancy = np.arange(1., 181.)
    data = {"unit_id": [7], "unit_data": {"hd_class": [2]},
            "occupancy_time_s": occupancy.tolist(), "spike_counts": [2 * occupancy]}
    monkeypatch.setattr(Path, "open", no_io)
    profiles = hd.hd_profiles(data, smoothing_deg=12, bins=30, hd_class=2)
    np.testing.assert_allclose(profiles, np.full((1, 30), 2), rtol=1e-14, atol=1e-14)


def test_ebb_profiles_keep_native_angles_nan_and_zero_without_io(monkeypatch):
    values = np.array([[1., 2.], [0., 0.], [np.nan, 3.], [np.nan, np.nan]])
    item = replace(asrfmap(values), unit_id=7, y_positions=np.array([3., 93., 183., 273.]),
                   x_positions=np.array([2., 8.]), metadata={"responseUnits": "Hz"})
    maps = RFMapList([item], Path("source_not_on_disk.rfmap"))
    original = maps.to_2d_array().copy()
    monkeypatch.setattr(Path, "open", no_io)

    profiles = ebc_profiles(maps, probe="B")

    np.testing.assert_array_equal(profiles, [[3., 0., np.nan, np.nan]])
    np.testing.assert_array_equal(profiles.columns, [3., 93., -177., -87.])
    assert profiles.index.tolist() == [("B", 7)]
    assert profiles.attrs == {"response_units": "Hz", "unit_info": {("B", 7): {}}}
    np.testing.assert_array_equal(maps.to_2d_array(), original)


def test_prepare_ebb_adds_keys_and_label_to_supplied_profiles(monkeypatch):
    values = np.tile([1., 2.], (60, 1))
    item = replace(asrfmap(values), unit_id=19, y_positions=np.arange(3., 360., 6.),
                   metadata={"responseUnits": "Hz"})
    maps = RFMapList([item], Path("source_not_on_disk.rfmap"))
    monkeypatch.setattr(Path, "open", no_io)

    table = prepare_ebc(maps, mouse="m22", date="261008", probe="A", label="EBC ring")

    assert table.shape == (1, 30)
    np.testing.assert_array_equal(table, np.full((1, 30), 3.))
    assert table.index.tolist() == [("m22", "261008", "A", 19)]
    assert table.attrs["label"] == "EBC ring"
    assert table.attrs["range"] == [-180, -90, 0, 90, 180]
    assert table.attrs["unit_info"] == {("m22", "261008", "A", 19): {}}


def test_explicit_basler_hd_rebuild_refreshes_existing_comparison(tmp_path, monkeypatch):
    from EBB import ebc_analysis as analysis

    session = tmp_path / "m22/261008/261008_11"
    comparison = session / "data/tc_comparison/hd_class3_ProbeA.csv"
    comparison.parent.mkdir(parents=True)
    comparison.write_text("previous comparison")
    monkeypatch.setattr(analysis, "read_formatted_json", lambda path: {"session_info": {}})
    monkeypatch.setattr(analysis, "get_exposure_timestamps", lambda *args, **kwargs: (
        np.arange(2) / 50, 0., {},
    ))
    feature = object()
    def make_tsd(exposures, frames, headings):
        np.testing.assert_array_equal(headings, [45., 135.])
        return feature, 50.
    monkeypatch.setattr(analysis, "make_head_direction_tsd", make_tsd)
    calls = []
    monkeypatch.setattr(analysis, "tuning_curve", lambda **kwargs: calls.append(("tuning", kwargs)))

    def generate(source, output, **settings):
        calls.append(("comparison", settings))
        assert calls[0][0] == "tuning"
        assert source == session / "data/tuning_curves/ProbeA/tuning_curves.tc"
        assert output == comparison
        assert settings == dict(bins=30, hd_class=3, unit_prefix="m22:261008:A", overwrite=True)
        output.write_text("fresh comparison")

    monkeypatch.setattr(analysis, "generate_hd_tc", generate)
    source = dict(kilosort_dir=str(session / "kilosort/ProbeA"), selected_interval_s=[0., .02],
                  adc_time_origin_s=0., camera_timing={"timestamp_reference": "test"},
                  pose_path=str(session / "261008.csv"))
    result = analysis.rebuild_basler_hd(dict(session=session, probe="A", frame_ids=np.arange(2),
                                            hd=np.array([44., 136.]), raw_hd_deg=np.array([45., 135.]),
                                            source=source, phase="baseline"),
                                      shuffles=23, seed=7)
    assert result == session / "data/tuning_curves/ProbeA/tuning_curves.tc"
    assert calls[0][1]["num_shuffle"] == 23
    assert calls[0][1]["shuffle_seed"] == 7
    assert calls[0][1]["HD_tsd"] is feature
    assert comparison.read_text() == "fresh comparison"
