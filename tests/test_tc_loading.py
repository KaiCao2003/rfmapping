"""Unified TC loading preserves RF CSV data and the existing HD reader path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils.rflocate import RFResult, detection, io


@pytest.fixture
def rf_csv(tmp_path):
    path = tmp_path / "chosen_responses.csv"
    path.write_text(
        "unit_id,-170.0,210.125,-12.5\n"
        "23,0,nan,1.2345678901234567\n"
        "7,nan,nan,nan\n"
        "99,2.345678901234567,0,0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def forbid_rf_analysis(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("RF CSV loading must not generate, detect, select, or resample profiles")

    for name in ("load_rfmap", "load_detected_rf", "load_hd_profiles",
                 "resample_profiles", "rf_pick", "select_rf_profiles"):
        monkeypatch.setattr(comparison, name, unexpected)
    monkeypatch.setattr(detection, "detect_rf", unexpected)
    monkeypatch.setattr(io, "save_rf_tc", unexpected)


def _assert_native_rf(table, *, recording=False):
    keys = [("B", unit_id) for unit_id in (23, 7, 99)]
    names = ["probe", "unit_id"]
    if recording:
        keys = [("m14", "260609", *key) for key in keys]
        names = ["mouse", "date", "probe", "unit_id"]
    assert table.index.names == names
    assert table.index.tolist() == keys
    np.testing.assert_array_equal(table.columns, [-170., 210.125, -12.5])
    np.testing.assert_array_equal(table.to_numpy(), [
        [0., np.nan, 1.2345678901234567],
        [np.nan, np.nan, np.nan],
        [2.345678901234567, 0., 0.],
    ])
    assert table.attrs["response_kind"] == "RF"
    assert table.attrs["response_units"] == "spike_count"
    assert table.attrs["unit_info"] == {
        key: {"zero_bins": zero} for key, zero in zip(keys, [1, 0, 2], strict=True)
    }


@pytest.mark.parametrize("recording", [False, True])
def test_load_tc_rf_preserves_native_data_and_optional_recording_keys(
    rf_csv, forbid_rf_analysis, recording,
):
    metadata = {"mouse": "m14", "date": 260609} if recording else {}
    before = rf_csv.read_bytes()
    table = comparison.load_tc(rf_csv, kind="RF", probe="B", **metadata)
    _assert_native_rf(table, recording=recording)
    assert table.attrs["label"] == "RF"
    assert table.attrs["source"] == str(rf_csv.resolve())
    assert rf_csv.read_bytes() == before
    assert list(rf_csv.parent.iterdir()) == [rf_csv]


def test_load_tc_rf_range_and_label_do_not_change_coordinates_or_values(rf_csv, forbid_rf_analysis):
    table = comparison.load_tc(
        rf_csv, kind="RF", mouse="m14", date=260609, probe="B",
        label="Chosen RF counts", range=[-180, 180],
    )
    _assert_native_rf(table, recording=True)
    assert table.attrs["label"] == "Chosen RF counts"
    assert table.attrs["range"] == [-180, 180]


@pytest.mark.parametrize("metadata", [{"mouse": "m14"}, {"date": 260609}])
def test_load_tc_requires_mouse_and_date_together(rf_csv, metadata):
    with pytest.raises(ValueError):
        comparison.load_tc(rf_csv, kind="RF", probe="B", **metadata)


@pytest.mark.parametrize("reader", ["profiles", "recording", "collection"])
def test_existing_rf_readers_accept_the_same_csv(rf_csv, forbid_rf_analysis, reader):
    if reader == "profiles":
        table = comparison.load_rf_profiles(rf_csv, probe="B")
    elif reader == "recording":
        table = comparison.load_rf(rf_csv, mouse="m14", date=260609, probe="B")
    else:
        collection = comparison.TuningCurveCollection()
        table = collection.load_tc(
            rf_csv, kind="RF", mouse="m14", date=260609, probe="B", prefix="RF3",
        )
        assert collection.profiles["RF3"] is table
    _assert_native_rf(table, recording=reader != "profiles")


def test_missing_rf_csv_raises_for_the_exact_path_without_fallback(tmp_path, forbid_rf_analysis):
    path = tmp_path / "missing_selected_rfonly.csv"
    with pytest.raises(FileNotFoundError) as exc:
        comparison.load_tc(path, kind="RF", mouse="m14", date=260609, probe="B")
    assert Path(exc.value.filename) == path
    assert list(tmp_path.iterdir()) == []


def test_omitted_kind_preserves_the_hd_loading_path(monkeypatch):
    index = pd.MultiIndex.from_tuples([("B", 23), ("B", 7)], names=["probe", "unit_id"])
    source = pd.DataFrame(np.arange(60, dtype=float).reshape(2, 30), index=index,
                          columns=np.arange(-174., 180., 12.))
    source.attrs["unit_info"] = {key: {"hd_class": 3} for key in index}
    calls = []

    def load_hd(path, **options):
        calls.append((path, options))
        return source.copy()

    monkeypatch.setattr(comparison, "load_hd_profiles", load_hd)
    default = comparison.load_tc("existing.tc", mouse="m14", date=260609, probe="B")
    explicit = comparison.load_tc("existing.tc", kind="HD", mouse="m14", date=260609, probe="B")
    pd.testing.assert_frame_equal(default, explicit)
    assert default.index.names == ["mouse", "date", "probe", "unit_id"]
    assert default.index.tolist() == [("m14", "260609", "B", 23), ("m14", "260609", "B", 7)]
    assert default.attrs["label"] == "HD"
    assert default.attrs["range"] == comparison.tcRange(False)
    assert calls == [(Path("existing.tc"), dict(probe="B"))] * 2
    comparison.load_tc("existing.tc", mouse="m14", date=260609, probe="B",
                       bins=60, smoothing_deg=4., hd_class=3)
    assert calls[-1] == (Path("existing.tc"), dict(probe="B", bins=60, smoothing_deg=4., hd_class=3))


@pytest.mark.parametrize("relative_path", [
    "m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeB/prepared.csv",
    "m14/260609/260609_3/data/regular_unitsSpikeCounts_260609_3_ProbeB_1d_rfonly.csv",
])
def test_load_tc_infers_recording_identity_and_probe_from_standard_paths(
    rf_csv, forbid_rf_analysis, relative_path, monkeypatch,
):
    path = rf_csv.parent / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(rf_csv.read_bytes())
    table = comparison.load_tc(path, kind="RF")
    _assert_native_rf(table, recording=True)
    assert table.attrs["source"] == str(path.resolve())
    monkeypatch.chdir(path.parent)
    relative = comparison.load_tc(path.name, kind="RF")
    pd.testing.assert_frame_equal(relative, table)
    assert relative.attrs == table.attrs


def test_explicit_recording_metadata_overrides_path_identity(rf_csv):
    path = rf_csv.parent / "m14/260609/260609_3/data/regular_ProbeB_1d.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(rf_csv.read_bytes())
    table = comparison.load_tc(path, kind="RF", mouse="m99", date=260101, probe="A")
    assert table.index.tolist() == [("m99", "260101", "A", unit_id) for unit_id in (23, 7, 99)]
    assert table.attrs["source"] == str(path.resolve())


def test_inferred_hd_and_rf_keys_match_biological_units_across_sessions(rf_csv, monkeypatch):
    root = rf_csv.parent / "m14/260609"
    hd_path = root / "260609_1/data/tuning_curves/ProbeB/tuning_curves.tc"
    rf_path = root / "260609_3/data/regular_unitsSpikeCounts_260609_3_ProbeB_1d.csv"
    rf_path.parent.mkdir(parents=True)
    rf_path.write_bytes(rf_csv.read_bytes())

    def load_hd(path, *, probe):
        assert path == hd_path and probe == "B"
        index = pd.MultiIndex.from_tuples([("B", 23), ("B", 99)], names=["probe", "unit_id"])
        table = pd.DataFrame(np.ones((2, 30)), index=index, columns=np.arange(-174., 180., 12.))
        table.attrs["unit_info"] = {key: {"hd_class": 3} for key in index}
        table.attrs["response_units"] = "Hz"
        return table

    monkeypatch.setattr(comparison, "load_hd_profiles", load_hd)
    hd = comparison.load_tc(hd_path)
    rf = comparison.load_tc(rf_path, kind="RF")
    assert hd.index.names == rf.index.names == ["mouse", "date", "probe", "unit_id"]
    assert hd.index.intersection(rf.index).tolist() == [
        ("m14", "260609", "B", 23), ("m14", "260609", "B", 99),
    ]
    assert hd.attrs["source"] == str(hd_path.resolve())
    assert rf.attrs["source"] == str(rf_path.resolve())
    assert hd.attrs["response_units"] == "Hz"


def test_concat_arbitrary_file_paths_namespaces_colliding_ids_and_keeps_selection_plotting(rf_csv):
    second_path = rf_csv.parent / "unrelated_curves.csv"
    second_path.write_bytes(rf_csv.read_bytes())
    first = comparison.load_tc(rf_csv, kind="RF")
    second = comparison.load_tc(second_path, kind="RF")
    assert first.index.names == second.index.names == ["probe", "unit_id"]
    assert first.index.tolist() == [("A", unit_id) for unit_id in (23, 7, 99)]
    combined = comparison.concat(first, second)
    assert combined.index.names == ["source", "probe", "unit_id"]
    expected_keys = [(str(path.resolve()), "A", unit_id)
                     for path in (rf_csv, second_path) for unit_id in (23, 7, 99)]
    assert combined.index.tolist() == expected_keys
    assert combined.attrs["unit_info"] == {
        key: {"zero_bins": zero} for key, zero in zip(expected_keys, [1, 0, 2] * 2, strict=True)
    }
    selected = comparison.rf_pick(combined, max_zero_bins=1)
    selected_keys = [key for key in expected_keys if key[-1] in (23, 7)]
    assert selected.index.tolist() == selected_keys
    assert set(selected.attrs["unit_info"]) == set(selected_keys)
    figure, axis = comparison.plot_profiles(selected, show=False)
    try:
        np.testing.assert_array_equal(
            np.ma.filled(axis.images[0].get_array(), np.nan), selected.to_numpy(),
        )
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            ":".join(map(str, key)) for key in selected_keys
        ]
        assert figure.axes[1].get_ylabel() == "Spike count"
    finally:
        plt.close(figure)
    assert first.index.names == ["probe", "unit_id"]


def test_concat_repeated_arbitrary_source_rejects_duplicate_keys(rf_csv):
    first = comparison.load_tc(rf_csv, kind="RF")
    same_source = comparison.load_tc(rf_csv.resolve(), kind="RF")
    with pytest.raises(ValueError):
        comparison.concat(first, same_source)


def test_nested_concat_preserves_source_keys_and_unit_metadata(rf_csv):
    paths = [rf_csv, rf_csv.parent / "second.csv", rf_csv.parent / "third.csv"]
    for path in paths[1:]:
        path.write_bytes(rf_csv.read_bytes())
    first, second, third = [comparison.load_tc(path, kind="RF") for path in paths]
    intermediate = comparison.concat(first, second)
    original_index = intermediate.index.copy()
    original_info = {key: value.copy() for key, value in intermediate.attrs["unit_info"].items()}
    nested = comparison.concat(intermediate, third)
    flat = comparison.concat(first, second, third)
    pd.testing.assert_frame_equal(nested, flat)
    assert nested.index.names == ["source", "probe", "unit_id"]
    assert nested.attrs["unit_info"] == flat.attrs["unit_info"]
    assert set(nested.attrs["unit_info"]) == set(nested.index)
    pd.testing.assert_index_equal(intermediate.index, original_index)
    assert intermediate.attrs["unit_info"] == original_info


def test_concat_loaded_rf_curves_preserves_recording_keys_and_native_values(rf_csv, forbid_rf_analysis):
    second_path = rf_csv.parent / "m15_curves.csv"
    second_path.write_text(
        "unit_id,-12.5,-170.0,210.125\n"
        "23,4,0,nan\n"
        "8,0,6,8\n",
        encoding="utf-8",
    )
    m14 = comparison.load_tc(rf_csv, kind="RF", mouse="m14", date=260609, probe="B")
    m15 = comparison.load_tc(second_path, kind="RF", mouse="m15", date=260630, probe="B")
    combined = comparison.concat(m14, m15, label="m14 + m15 RF")
    assert combined.index.names == ["mouse", "date", "probe", "unit_id"]
    assert combined.index.tolist() == [
        ("m14", "260609", "B", 23), ("m14", "260609", "B", 7),
        ("m14", "260609", "B", 99), ("m15", "260630", "B", 23),
        ("m15", "260630", "B", 8),
    ]
    np.testing.assert_array_equal(combined.columns, [-170., 210.125, -12.5])
    np.testing.assert_array_equal(combined.to_numpy(), [
        [0., np.nan, 1.2345678901234567], [np.nan, np.nan, np.nan],
        [2.345678901234567, 0., 0.], [0., np.nan, 4.], [6., 8., 0.],
    ])
    assert combined.attrs["label"] == "m14 + m15 RF"
    assert combined.attrs["response_kind"] == "RF"
    assert combined.attrs["unit_info"] == {**m14.attrs["unit_info"], **m15.attrs["unit_info"]}
    np.testing.assert_array_equal(m15.columns, [-12.5, -170., 210.125])


@pytest.mark.parametrize("response_units, colorbar_label", [
    (None, "Spike count"), ("Hz", "Firing rate (Hz)"),
])
def test_plot_profiles_uses_tc_metadata_and_preserves_supplied_values(
    rf_csv, forbid_rf_analysis, monkeypatch, response_units, colorbar_label,
):
    table = comparison.load_tc(rf_csv, kind="RF", probe="B", response_units=response_units)
    original = table.copy(deep=True)
    monkeypatch.setattr(plt, "show", lambda: None)
    figure, axis = comparison.plot_profiles(table)
    try:
        assert axis.get_title() == "RF"
        assert axis.get_xlabel() == "RF direction (°)"
        assert axis.get_ylabel() == "Unit ID"
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["B:23", "B:7", "B:99"]
        assert figure.axes[1].get_ylabel() == colorbar_label
        np.testing.assert_array_equal(
            np.ma.filled(axis.images[0].get_array(), np.nan),
            [[0., np.nan, 1.2345678901234567], [np.nan, np.nan, np.nan],
             [2.345678901234567, 0., 0.]],
        )
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs
    finally:
        plt.close(figure)


@pytest.mark.parametrize("transform, response_units, colorbar_label", [
    (comparison.normalize_tc, "normalized", "Normalized response"),
    (comparison.zscore_tc, "zscore", "Z-score (SD)"),
])
def test_explicit_tc_transforms_update_only_the_result_unit_metadata(
    rf_csv, transform, response_units, colorbar_label,
):
    raw = comparison.load_tc(rf_csv, kind="RF", probe="B")
    transformed = transform(raw)
    assert transformed.attrs["response_units"] == response_units
    assert raw.attrs["response_units"] == "spike_count"
    assert transformed.attrs["label"] == raw.attrs["label"] == "RF"
    figure, axis = comparison.plot_profiles(transformed, show=False)
    try:
        assert figure.axes[1].get_ylabel() == colorbar_label
        np.testing.assert_array_equal(
            np.ma.filled(axis.images[0].get_array(), np.nan), transformed.to_numpy(),
        )
    finally:
        plt.close(figure)


def test_concat_mixed_units_joins_labels_without_claiming_a_shared_rate_unit(rf_csv):
    counts = comparison.load_tc(rf_csv, kind="RF", mouse="m14", date=260609,
                                 probe="B", label="RF counts")
    rates = comparison.load_tc(rf_csv, kind="RF", mouse="m15", date=260630,
                                probe="B", label="RF rate", response_units="Hz")
    combined = comparison.concat(counts, rates)
    assert combined.attrs["label"] == "RF counts + RF rate"
    assert "response_units" not in combined.attrs
    assert combined.attrs["response_kind"] == "RF"
    np.testing.assert_array_equal(combined.to_numpy(), np.vstack([counts, rates]))
    figure, axis = comparison.plot_profiles(combined, show=False)
    try:
        assert figure.axes[1].get_ylabel() == "Response"
        assert axis.get_xlabel() == "RF counts + RF rate direction (°)"
    finally:
        plt.close(figure)


def test_concat_explicit_range_accepts_loaded_rf_csv_without_range_metadata(tmp_path):
    centers = np.arange(-174., 180., 12.)
    values = np.vstack([np.arange(30, dtype=float), np.zeros(30)])
    values[0, 7] = np.nan
    values[1, 15] = 3.
    path = tmp_path / "native_thirty_bins.csv"
    pd.DataFrame(values, index=pd.Index([23, 7], name="unit_id"), columns=centers).to_csv(path)
    m14 = comparison.load_tc(path, kind="RF", mouse="m14", date=260609, probe="B")
    m15 = comparison.load_tc(path, kind="RF", mouse="m15", date=260630, probe="B")
    assert "range" not in m14.attrs and "range" not in m15.attrs

    explicit = [comparison.resample_profiles(table, range=[-180, 180]) for table in (m14, m15)]
    expected = pd.concat(explicit)
    combined = comparison.concat(m14, m15, range=[-180, 180])
    pd.testing.assert_frame_equal(combined, expected)
    np.testing.assert_array_equal(combined.columns, centers)
    np.testing.assert_array_equal(combined.to_numpy(), np.vstack([values, values]))
    assert combined.attrs["range"] == [-180, 180]
    assert "range" not in m14.attrs and "range" not in m15.attrs


def _detected_rf():
    masks = np.array([[[0]], [[1]], [[1]]], dtype=np.uint8)
    return RFResult(masks, masks, np.array([99, 23, 7]), {}, "selection")


def test_select_rf_profiles_accepts_one_full_recording_group(rf_csv):
    profiles = comparison.load_tc(rf_csv, kind="RF", mouse="m14", date=260609, probe="B")
    selected = comparison.select_rf_profiles(profiles, _detected_rf())
    expected_keys = [("m14", "260609", "B", 23), ("m14", "260609", "B", 7)]
    pd.testing.assert_frame_equal(selected, profiles.loc[expected_keys])
    assert selected.attrs["unit_info"] == {
        key: profiles.attrs["unit_info"][key] for key in expected_keys
    }


@pytest.mark.parametrize("field, different", [("mouse", "m15"), ("date", "260610"), ("probe", "A")])
def test_select_rf_profiles_rejects_multiple_recording_groups(rf_csv, field, different):
    first = comparison.load_tc(rf_csv, kind="RF", mouse="m14", date=260609, probe="B")
    metadata = {"mouse": "m14", "date": 260609, "probe": "B", field: different}
    second = comparison.load_tc(rf_csv, kind="RF", **metadata)
    pooled = pd.concat([first, second])
    pooled.attrs["unit_info"] = {**first.attrs["unit_info"], **second.attrs["unit_info"]}
    with pytest.raises(ValueError):
        comparison.select_rf_profiles(pooled, _detected_rf())
