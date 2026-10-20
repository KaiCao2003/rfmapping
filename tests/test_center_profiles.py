"""CB/CD preparation preserves occupancy weighting, identity and linear distance."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from EBB import center_profiles
from EBB.center_profiles import (
    cb_profiles, cd_profiles, generate_center_tc, load_center_comparison,
    prep_cb_tc, prep_cd_tc, prepare_cb, prepare_cd,
)


def tuning(feature):
    return {
        "metadata": {"feature": feature, "coordinate_convention": "native clockwise" if feature == "cb" else "distance cm"},
        "unit_id": [23, 0, 7],
        "angle_bin_edges_deg" if feature == "cb" else "distance_bin_edges_cm":
            [0., 90., 180., 270., 360.] if feature == "cb" else [0., 100., 200., 300., 400.],
        "occupancy_time_s": [1., 3., 0., 2.],
        "spike_counts": [[2, 18, 0, 8], [0, 0, 0, 0], [4, 0, 0, 2]],
        "unit_data": {f"{feature}_is_cell": [True, False, None],
                      f"{feature}_status": ["classified", "classified", "shuffle_not_run"]},
    }


@pytest.mark.parametrize("feature,profiles", [("cb", cb_profiles), ("cd", cd_profiles)])
def test_native_profiles_keep_raw_rates_zero_missing_and_saved_classes(feature, profiles):
    data = tuning(feature)
    original = copy.deepcopy(data)
    table = profiles(data, probe="B")
    assert table.index.tolist() == [("B", 23), ("B", 0), ("B", 7)]
    np.testing.assert_allclose(table.iloc[0], [2., 6., np.nan, 4.], equal_nan=True)
    np.testing.assert_allclose(table.iloc[1], [0., 0., np.nan, 0.], equal_nan=True)
    np.testing.assert_array_equal(table.columns, [45., 135., 225., 315.] if feature == "cb" else [50., 150., 250., 350.])
    assert table.attrs["unit_info"][("B", 7)][f"{feature}_is_cell"] is None
    selected = profiles(data, cells_only=True)
    assert selected.index.tolist() == [("A", 23)]
    assert set(selected.attrs["unit_info"]) == {("A", 23)}
    assert data == original


@pytest.mark.parametrize("feature,profiles", [("cb", cb_profiles), ("cd", cd_profiles)])
def test_profile_rebinning_sums_counts_and_occupancy_before_division(feature, profiles):
    table = profiles(tuning(feature), probe=None, bins=2)
    assert table.index.tolist() == [23, 0, 7]
    np.testing.assert_allclose(table.iloc[0], [5., 4.])
    np.testing.assert_array_equal(table.iloc[1], [0., 0.])
    np.testing.assert_array_equal(table.columns, [90., 270.] if feature == "cb" else [100., 300.])
    assert table.attrs["response_units"] == "Hz"
    assert table.attrs["coordinate_units"] == ("degrees" if feature == "cb" else "cm")


def test_comparison_preparation_never_wraps_distance_or_interpolates_unvisited_bins():
    raw = cd_profiles(tuning("cd"))
    native = prep_cd_tc(raw)
    np.testing.assert_array_equal(native.columns, [50., 150., 250., 350.])
    np.testing.assert_allclose(native.iloc[0], [1 / 3, 1., np.nan, 2 / 3], equal_nan=True)
    assert native.iloc[1, 0] == 0.
    assert np.isnan(native.iloc[1, 2])
    rebinned = prep_cd_tc(raw, bins=2)
    np.testing.assert_array_equal(rebinned.columns, [100., 300.])
    np.testing.assert_allclose(rebinned.iloc[0], [1., np.nan], equal_nan=True)
    assert rebinned.attrs["coordinate_units"] == "cm"
    assert "range" not in rebinned.attrs
    pd.testing.assert_frame_equal(raw, cd_profiles(tuning("cd")))


def test_cb_preparation_retains_clockwise_order_across_signed_boundary():
    table = cb_profiles(tuning("cb"))
    table.columns = [45., 135., -135., -45.]
    prepared = prep_cb_tc(table, bins=2)
    np.testing.assert_array_equal(prepared.columns, [90., 270.])
    np.testing.assert_allclose(prepared.iloc[0], [1., np.nan], equal_nan=True)
    assert prepared.attrs["response_units"] == "normalized"


@pytest.mark.parametrize("feature,prepare", [("cb", prepare_cb), ("cd", prepare_cd)])
def test_prepare_adds_recording_identity_and_preserves_unit_metadata(feature, prepare):
    table = prepare(tuning(feature), mouse="m22", date="261008", probe="B", bins=2, cells_only=True)
    assert table.index.names == ["mouse", "date", "probe", "unit_id"]
    assert table.index.tolist() == [("m22", "261008", "B", 23)]
    assert set(table.attrs["unit_info"]) == {("m22", "261008", "B", 23)}
    assert table.attrs["label"] == feature.upper()
    np.testing.assert_allclose(table.iloc[0], [5., 4.])


@pytest.mark.parametrize("feature", ["cb", "cd"])
def test_explicit_comparison_export_reuses_saved_classes_and_preserves_source(tmp_path, feature):
    source = tmp_path / f"{feature}.tc"
    source.write_text(json.dumps(tuning(feature)))
    original = source.read_bytes()
    output = tmp_path / "comparison" / f"{feature}.csv"
    returned = generate_center_tc(source, output, feature=feature, unit_prefix="m22:261008:B",
                                  bins=2, cells_only=True)
    assert returned == output
    table = load_center_comparison(output, feature=feature)
    assert table.index.tolist() == ["m22:261008:B:23"]
    np.testing.assert_allclose(table.iloc[0], [5., 4.])
    np.testing.assert_array_equal(table.columns, [90., 270.] if feature == "cb" else [100., 300.])
    assert table.attrs["response_units"] == "Hz"
    assert source.read_bytes() == original
    source.unlink()
    pd.testing.assert_frame_equal(load_center_comparison(output, feature=feature), table)


def test_existing_comparison_requires_overwrite_before_source_is_read(tmp_path, monkeypatch):
    output = tmp_path / "saved.csv"
    output.write_text("unit_id,1.0\nexisting:0,0\n")
    before = output.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("Existing output must be protected before reading source")

    monkeypatch.setattr(center_profiles, "load_center_tc", unexpected)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        generate_center_tc(tmp_path / "missing.tc", output, feature="cd", unit_prefix="paired")
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="every input"):
        generate_center_tc(output, output, feature="cb", unit_prefix="paired", overwrite=True)


@pytest.mark.parametrize("bins", [0, -1, True, 1.5, 3, 5])
def test_profiles_reject_incompatible_rebinning(bins):
    with pytest.raises(ValueError, match="dividing"):
        cb_profiles(tuning("cb"), bins=bins)


def test_center_reader_does_not_generate_missing_comparison(tmp_path):
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_center_comparison(path, feature="cd")
    assert not path.exists()
