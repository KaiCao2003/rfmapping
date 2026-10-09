"""TC CSV loading and concatenation preserve supplied identity and native data."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils.rflocate import RFResult, detection, io


@pytest.fixture
def tc_csv(tmp_path):
    path = tmp_path / "chosen_responses.csv"
    path.write_text(
        "unit_id,-170.0,210.125,-12.5\n"
        "m14:260609:B:23,0,nan,1.2345678901234567\n"
        "007,nan,nan,nan\n"
        "NA,2.345678901234567,0,0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rf_csv(tmp_path):
    path = tmp_path / "native_rf.csv"
    path.write_text(
        "unit_id,-170.0,210.125,-12.5\n23,0,nan,1\n7,nan,nan,nan\n99,2,0,0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def forbid_csv_analysis(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CSV loading and concat must not read raw sources, analyze, or transform curves")

    for name in ("load_rfmap", "load_rf_tc", "load_rf_profiles", "rf_profiles",
                 "load_detected_rf", "load_hd_profiles", "load_ebc_profiles",
                 "read_formatted_json", "update_hd_classification", "recording_profiles",
                 "resample_profiles", "_resample_profiles", "_interpolate_angles",
                 "normalize_tc", "zscore_tc", "smooth_profiles", "hd_pick", "rf_pick",
                 "select_rf_profiles"):
        monkeypatch.setattr(comparison, name, unexpected)
    monkeypatch.setattr(detection, "detect_rf", unexpected)
    monkeypatch.setattr(io, "save_rf_tc", unexpected)


def _assert_native_csv(table):
    assert table.index.names == ["unit_id"]
    assert table.index.tolist() == ["m14:260609:B:23", "007", "NA"]
    np.testing.assert_array_equal(table.columns, [-170., 210.125, -12.5])
    np.testing.assert_array_equal(table.to_numpy(), [
        [0., np.nan, 1.2345678901234567],
        [np.nan, np.nan, np.nan],
        [2.345678901234567, 0., 0.],
    ])


def test_load_tc_preserves_opaque_ids_and_native_data_without_guessed_metadata(
    tc_csv, forbid_csv_analysis,
):
    before = tc_csv.read_bytes()
    table = comparison.tc_loader(tc_csv)
    _assert_native_csv(table)
    assert table.attrs == {}
    assert tc_csv.read_bytes() == before
    assert list(tc_csv.parent.iterdir()) == [tc_csv]


def test_load_tc_metadata_only_describes_values(tc_csv, forbid_csv_analysis):
    table = comparison.tc_loader(tc_csv, label="Chosen counts", range=[-180, 180],
                                 response_units="spike_count")
    _assert_native_csv(table)
    assert table.attrs == {
        "label": "Chosen counts", "range": [-180, 180], "response_units": "spike_count",
    }


@pytest.mark.parametrize("relative_path", [
    "m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeB/prepared.csv",
    "m99/260101/260101_1/data/regular_ProbeA_1d.csv",
    "renamed.csv",
])
def test_relocation_and_filename_never_change_identity(tc_csv, forbid_csv_analysis,
                                                       relative_path, monkeypatch):
    path = tc_csv.parent / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tc_csv.read_bytes())
    original = comparison.tc_loader(tc_csv)
    relocated = comparison.tc_loader(path)
    pd.testing.assert_frame_equal(relocated, original)
    assert relocated.attrs == original.attrs
    monkeypatch.chdir(path.parent)
    pd.testing.assert_frame_equal(comparison.tc_loader(path.name), original)


@pytest.mark.parametrize("option", [
    {"kind": "RF"}, {"probe": "B"}, {"mouse": "m14", "date": 260609}, {"bins": 30},
])
def test_raw_reader_and_identity_options_are_not_supported(tc_csv, option):
    with pytest.raises(TypeError):
        comparison.tc_loader(tc_csv, **option)


def test_missing_csv_raises_for_exact_path_without_fallback(tmp_path, forbid_csv_analysis):
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError) as exc:
        comparison.tc_loader(path)
    assert Path(exc.value.filename) == path
    assert list(tmp_path.iterdir()) == []


def test_load_tc_does_not_dispatch_raw_files(tmp_path, forbid_csv_analysis):
    path = tmp_path / "tuning_curves.tc"
    path.write_text('{"unit_id": [7], "spike_counts": [[1, 2]]}', encoding="utf-8")
    with pytest.raises(KeyError):
        comparison.tc_loader(path)


def test_csv_exact_round_trip_preserves_complete_ids_values_and_angles(tmp_path, forbid_csv_analysis):
    index = pd.Index(["m14:260609:B:23", "m15:260630:B:23", "007"], name="unit_id")
    angles = [-170., -12.5, 190.12345678912345]
    values = np.array([
        [0., np.nan, 1.2345678901234567],
        [2.345678901234567, 0., 0.0000001234567891234567],
        [np.nan, np.nan, np.nan],
    ])
    table = pd.DataFrame(values, index=index, columns=angles)
    table.attrs = {"label": "Not serialized", "unit_info": {"not": "exported"}}
    original = table.copy(deep=True)
    path = tmp_path / "prepared.csv"
    comparison.save_tc(table, path)
    loaded = comparison.tc_loader(path)
    np.testing.assert_array_equal(loaded.to_numpy(), values)
    pd.testing.assert_index_equal(loaded.index, index)
    np.testing.assert_array_equal(loaded.columns, angles)
    assert loaded.attrs == {}
    pd.testing.assert_frame_equal(table, original)
    assert table.attrs == original.attrs
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["unit_id", *map(str, angles)]
    assert [row[0] for row in rows[1:]] == index.tolist()
    assert len(rows) == 4 and all(len(row) == 4 for row in rows)
    assert list(tmp_path.iterdir()) == [path]


def test_save_tc_rejects_multiindex_instead_of_discarding_identity(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [("m14", "260609", "A", 7), ("m15", "260630", "B", 8)],
        names=["mouse", "date", "probe", "unit_id"],
    )
    table = pd.DataFrame([[1., 2.], [3., 4.]], index=index, columns=[-90., 90.])
    path = tmp_path / "pooled.csv"
    with pytest.raises(ValueError, match="simple index"):
        comparison.save_tc(table, path)
    assert not path.exists()


@pytest.mark.parametrize("ids", [["same", "same"], [7, "7"]])
def test_save_tc_rejects_duplicate_serialized_ids(tmp_path, ids):
    table = pd.DataFrame([[1.], [2.]], index=ids, columns=[0.])
    with pytest.raises(ValueError, match="unit IDs must be unique"):
        comparison.save_tc(table, tmp_path / "duplicate.csv")
    assert list(tmp_path.iterdir()) == []


def test_concat_only_stacks_and_reorders_matching_degrees(tc_csv, forbid_csv_analysis):
    first = comparison.tc_loader(tc_csv, label="First", response_units="Hz")
    second = pd.DataFrame([[4., 0., np.nan], [0., 6., 8.]],
                          index=pd.Index(["m15:260630:B:23", "m15:260630:B:8"], name="unit_id"),
                          columns=[-12.5, -170., 210.125])
    second.attrs = {"label": "Second", "response_units": "Hz"}
    combined = comparison.concat(first, second, label="Combined")
    assert combined.index.tolist() == [*first.index, *second.index]
    np.testing.assert_array_equal(combined.columns, first.columns)
    np.testing.assert_array_equal(combined.to_numpy(), [
        [0., np.nan, 1.2345678901234567], [np.nan, np.nan, np.nan],
        [2.345678901234567, 0., 0.], [0., np.nan, 4.], [6., 8., 0.],
    ])
    assert combined.attrs == {"label": "Combined", "response_units": "Hz"}
    np.testing.assert_array_equal(second.columns, [-12.5, -170., 210.125])


def test_concat_does_not_disambiguate_duplicates_by_source_path(tc_csv, forbid_csv_analysis):
    first = comparison.tc_loader(tc_csv)
    second = first.copy()
    first.attrs["source"] = "/first.csv"
    second.attrs["source"] = "/second.csv"
    with pytest.raises(ValueError):
        comparison.concat(first, second)


def test_concat_rejects_different_grids_without_transforming(tc_csv, forbid_csv_analysis):
    first = comparison.tc_loader(tc_csv)
    second = pd.DataFrame([[1., 2.]], index=["another"], columns=[-90., 90.])
    with pytest.raises(ValueError, match="resample_profiles explicitly"):
        comparison.concat(first, second)
    with pytest.raises(TypeError):
        comparison.concat(first, second, range=[-180, 180])
    assert comparison.combine is comparison.concat


def test_collection_load_and_combine_only_register_generic_csvs(tc_csv, forbid_csv_analysis):
    collection = comparison.TuningCurveCollection()
    table = collection.load_tc(tc_csv, prefix="chosen", label="Chosen", response_units="Hz")
    _assert_native_csv(table)
    assert collection.profiles["chosen"] is table
    assert table.attrs == {"label": "Chosen", "response_units": "Hz"}
    result = collection.combine("pooled", ["chosen"])
    pd.testing.assert_frame_equal(result, table)
    assert collection.profiles["pooled"] is result


def test_concat_merges_only_explicit_metadata():
    first = pd.DataFrame([[1., 2.]], index=["a"], columns=[-90., 90.])
    second = pd.DataFrame([[3., 4.]], index=["b"], columns=[-90., 90.])
    third = pd.DataFrame([[5., 6.]], index=["c"], columns=[-90., 90.])
    for table, label, units in ((first, "Counts", "spike_count"),
                                (second, "Rates", "Hz"), (third, "Third", "Hz")):
        table.attrs = {"label": label, "response_units": units,
                       "unit_info": {table.index[0]: {"selected": True}}}
    intermediate = comparison.concat(first, second)
    assert "response_units" not in intermediate.attrs
    nested = comparison.concat(intermediate, third)
    flat = comparison.concat(first, second, third)
    pd.testing.assert_frame_equal(nested, flat)
    assert nested.attrs == flat.attrs
    assert nested.attrs["label"] == "Counts + Rates + Third"
    assert set(nested.attrs["unit_info"]) == set(nested.index)
    assert set(intermediate.attrs["unit_info"]) == {"a", "b"}


def test_rf_pick_uses_supplied_native_values_and_does_not_need_cached_qc(tc_csv):
    table = comparison.tc_loader(tc_csv)
    selected = comparison.rf_pick(table, max_zero_bins=1)
    assert selected.index.tolist() == ["m14:260609:B:23", "007"]
    assert selected.attrs == {}
    assert len(comparison.rf_pick(table, max_zero_bins=None)) == 3
    table.attrs["unit_info"] = {key: {"zero_bins": 99} for key in table.index}
    selected = comparison.rf_pick(table, max_zero_bins=1)
    assert selected.index.tolist() == ["m14:260609:B:23", "007"]
    assert set(selected.attrs["unit_info"]) == set(selected.index)


@pytest.mark.parametrize("response_units, colorbar_label", [
    (None, "Response"), ("Hz", "Firing rate (Hz)"), ("spike_count", "Spike count"),
])
def test_plot_profiles_uses_only_explicit_units(tc_csv, forbid_csv_analysis,
                                               response_units, colorbar_label):
    table = comparison.tc_loader(tc_csv, label="Chosen", response_units=response_units)
    original = table.copy(deep=True)
    figure, axis = comparison.plot_profiles(table, show=False)
    try:
        assert axis.get_title() == "Chosen"
        assert axis.get_xlabel() == "Chosen direction (°)"
        assert [tick.get_text() for tick in axis.get_yticklabels()] == table.index.tolist()
        assert figure.axes[1].get_ylabel() == colorbar_label
        np.testing.assert_array_equal(np.ma.filled(axis.images[0].get_array(), np.nan), table.to_numpy())
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs
    finally:
        plt.close(figure)


@pytest.mark.parametrize("transform, response_units, colorbar_label", [
    (comparison.normalize_tc, "normalized", "Normalized response"),
    (comparison.zscore_tc, "zscore", "Z-score (SD)"),
])
def test_explicit_transforms_update_result_units_only(tc_csv, transform, response_units, colorbar_label):
    raw = comparison.tc_loader(tc_csv, label="Chosen", response_units="spike_count")
    transformed = transform(raw)
    assert transformed.attrs["response_units"] == response_units
    assert raw.attrs["response_units"] == "spike_count"
    figure, _ = comparison.plot_profiles(transformed, show=False)
    try:
        assert figure.axes[1].get_ylabel() == colorbar_label
    finally:
        plt.close(figure)


@pytest.mark.parametrize("recording", [False, True])
def test_existing_rf_readers_keep_explicit_recording_identity(rf_csv, recording):
    table = (comparison.load_rf(rf_csv, mouse="m14", date=260609, probe="B") if recording else
             comparison.load_rf_profiles(rf_csv, probe="B"))
    keys = [("B", unit_id) for unit_id in (23, 7, 99)]
    if recording:
        keys = [("m14", "260609", *key) for key in keys]
    assert table.index.tolist() == keys
    np.testing.assert_array_equal(table.columns, [-170., 210.125, -12.5])
    assert table.attrs["response_kind"] == "RF"
    assert table.attrs["response_units"] == "spike_count"
    assert [info["zero_bins"] for info in table.attrs["unit_info"].values()] == [1, 0, 2]


def _detected_rf():
    masks = np.array([[[0]], [[1]], [[1]]], dtype=np.uint8)
    return RFResult(masks, masks, np.array([99, 23, 7]), {}, "selection")


def test_select_rf_profiles_accepts_one_full_recording_group(rf_csv):
    profiles = comparison.load_rf(rf_csv, mouse="m14", date=260609, probe="B")
    selected = comparison.select_rf_profiles(profiles, _detected_rf())
    expected_keys = [("m14", "260609", "B", 23), ("m14", "260609", "B", 7)]
    pd.testing.assert_frame_equal(selected, profiles.loc[expected_keys])
    assert selected.attrs["unit_info"] == {
        key: profiles.attrs["unit_info"][key] for key in expected_keys
    }


@pytest.mark.parametrize("field, different", [("mouse", "m15"), ("date", "260610"), ("probe", "A")])
def test_select_rf_profiles_rejects_multiple_recording_groups(rf_csv, field, different):
    first = comparison.load_rf(rf_csv, mouse="m14", date=260609, probe="B")
    metadata = {"mouse": "m14", "date": 260609, "probe": "B", field: different}
    second = comparison.load_rf(rf_csv, **metadata)
    pooled = comparison.concat(first, second)
    with pytest.raises(ValueError):
        comparison.select_rf_profiles(pooled, _detected_rf())
