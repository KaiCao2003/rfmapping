"""Prepare missing comparison CSVs without changing saved scientific inputs."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import locate_rf
from Utils import direction_comparison as comparison
from Utils import tc_preparation as preparation
from Utils.direction_comparison import load_tc
from Utils.rflocate import RFResult, load_rf_tc, save_rf
from Utils.rflocate import _detector


@pytest.fixture
def rf_inputs(tmp_path):
    source = tmp_path / "source.rfmap"
    grid = np.array([[1, 2], [10, 20], [3, 4]])
    responses = np.stack([grid, np.zeros_like(grid), grid * 2])
    counts = np.stack([responses, responses * 10, responses * 100], axis=-1)
    source.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7, 11, 23], "xPositions": [-145., -135.],
        "yPositions": [-20., 0., 20.], "timeBinEdges": [0., .1, .2, .3],
    }))
    masks = np.array([
        [[1, 0], [0, 0], [0, 1]],
        [[0, 0], [1, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ], dtype=np.uint8)
    centers = np.zeros_like(masks)
    centers[0, 0, 0] = centers[1, 1, 0] = 1
    detected = tmp_path / "saved_detection.npz"
    save_rf(RFResult(masks, centers, np.array([23, 7, 11]), {}, "supplied"), detected)
    return source, detected


def _unexpected(*args, **kwargs):
    pytest.fail("Existing outputs must not read sources or rerun classification/detection")


@pytest.mark.parametrize("rf_only", [False, True])
def test_missing_rf_comparison_generates_counts_then_selects_saved_units(
    tmp_path, monkeypatch, rf_inputs, rf_only,
):
    source, detected = rf_inputs
    original = {path: path.read_bytes() for path in rf_inputs}
    monkeypatch.setattr(_detector, "detect_rf", _unexpected)
    projection = tmp_path / "native" / "x.csv"
    output = tmp_path / "comparison" / "rf.csv"
    returned = preparation.prepare_rf_comparison(
        source, output, projection_path=projection, detected_rf_path=detected,
        time_range_s=(0., .2), rf_only=rf_only, unit_prefix="mouse:day:A",
    )

    assert returned == output
    native = load_rf_tc(projection)
    actual = load_tc(output)
    expected_ids = [23, 7] if rf_only else [7, 23]
    expected_values = [[88, 132], [110, 220]] if rf_only else [[154, 286], [308, 572]]
    assert actual.index.tolist() == [f"mouse:day:A:{unit}" for unit in expected_ids]
    np.testing.assert_array_equal(actual.columns, [-145., -135.])
    np.testing.assert_array_equal(actual.to_numpy(), expected_values)
    np.testing.assert_array_equal(actual.to_numpy(), native.loc[expected_ids].to_numpy())
    if rf_only:
        assert native.loc[11].isna().all()
    else:
        assert native.loc[11].eq(0).all()
    assert {path: path.read_bytes() for path in rf_inputs} == original


def test_existing_native_projection_is_reused_without_raw_map(tmp_path, monkeypatch, rf_inputs):
    source, detected = rf_inputs
    source.unlink()
    projection = tmp_path / "native.csv"
    projection.write_text(
        "unit_id,210.125,-60.0,0.0\n"
        "7,0,nan,1.2345678901234567\n"
        "11,8,9,10\n"
        "23,2.345678901234567,0,0\n",
    )
    before = projection.read_bytes()
    monkeypatch.setattr(preparation, "load_rfmap", _unexpected)
    monkeypatch.setattr(_detector, "detect_rf", _unexpected)
    output = tmp_path / "comparison" / "rf.csv"
    preparation.prepare_rf_comparison(
        source, output, projection_path=projection, detected_rf_path=detected,
        time_range_s=(0., .2), rf_only=True, unit_prefix="paired",
    )
    expected = load_rf_tc(projection).loc[[7, 23]]
    expected.index = pd.Index(["paired:7", "paired:23"], name="unit_id")
    pd.testing.assert_frame_equal(load_tc(output), expected)
    assert projection.read_bytes() == before
    assert not source.exists()


@pytest.mark.parametrize("kind", ["hd", "rf", "projection", "comparison"])
def test_existing_csv_returns_before_reading_any_source(tmp_path, monkeypatch, kind):
    output = tmp_path / "saved.csv"
    output.write_text("unit_id,0.0,12.0\nexisting:23,0,nan\n")
    before = output.read_bytes()
    for function in ("load_hd_profiles", "load_rfmap", "load_rf", "load_rf_tc"):
        monkeypatch.setattr(preparation, function, _unexpected)
    monkeypatch.setattr(comparison, "update_hd_classification", _unexpected)
    monkeypatch.setattr(_detector, "detect_rf", _unexpected)
    absent = tmp_path / "absent_source"
    if kind == "hd":
        returned = preparation.prepare_hd_tc(absent, output, bins=30, hd_class=3, unit_prefix="unused")
    elif kind == "rf":
        returned = preparation.prepare_rf_tc(absent, output, select_results=[absent], unit_prefix="unused")
    else:
        options = dict(detected_rf_path=absent, time_range_s=(0., .2), rf_only=True)
        if kind == "projection":
            returned = preparation.prepare_rf_projection(absent, output, **options)
        else:
            returned = preparation.prepare_rf_comparison(
                absent, output, projection_path=absent, unit_prefix="unused", **options,
            )
    assert returned == output
    assert output.read_bytes() == before
    assert set(tmp_path.iterdir()) == {output}


@pytest.mark.parametrize("hd_class", [None, 3])
def test_missing_hd_csv_rebins_unsmoothed_counts_and_selects_class(tmp_path, hd_class):
    source = tmp_path / "source.tc"
    edges = np.linspace(0, 360, 181)
    angles = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    rates = np.array([np.exp(.6 * np.cos(angles)), np.ones(180), np.exp(.6 * np.cos(angles))])
    occupancy = np.linspace(1, 3, 180)
    counts = rates * occupancy
    source.write_text(json.dumps({
        "metadata": {"classification": {}}, "unit_id": [23, 7, 99],
        "angle_bin_edges_deg": edges.tolist(), "occupancy_time_s": occupancy.tolist(),
        "firing_rate_hz": rates.tolist(), "spike_counts": counts.tolist(),
        "unit_data": {"rayleigh_p": [.001, .001, .2], "shuffle_p": [.001, .001, .001]},
    }))
    before = source.read_bytes()
    output = tmp_path / "comparison" / "hd.csv"
    returned = preparation.prepare_hd_tc(
        source, output, bins=6, hd_class=hd_class, unit_prefix="paired",
    )
    actual = load_tc(output)
    expected = counts.reshape(3, 6, 30).sum(axis=2) / occupancy.reshape(6, 30).sum(axis=1)
    expected_ids = ["paired:23", "paired:7", "paired:99"]
    if hd_class is not None:
        expected, expected_ids = expected[:1], expected_ids[:1]
    assert returned == output
    assert actual.index.tolist() == expected_ids
    np.testing.assert_array_equal(actual.columns, [30., 90., 150., -150., -90., -30.])
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    assert source.read_bytes() == before


def test_rf_preparation_overwrite_is_explicit_and_retains_native_values(tmp_path):
    source = tmp_path / "native.csv"
    source.write_text("unit_id,210.125,-60.0,0.0\n23,0,nan,1.2345678901234567\n7,8,9,10\n")
    output = tmp_path / "comparison" / "rf.csv"
    preparation.prepare_rf_tc(source, output, unit_prefix="first")
    first = output.read_bytes()
    preparation.prepare_rf_tc(source, output, unit_prefix="second")
    assert output.read_bytes() == first
    preparation.prepare_rf_tc(source, output, unit_prefix="second", overwrite=True)
    expected = load_rf_tc(source).astype(float)
    expected.index = pd.Index(["second:23", "second:7"], name="unit_id")
    pd.testing.assert_frame_equal(load_tc(output), expected)


@pytest.mark.parametrize("output_input", ["source", "projection", "detected"])
def test_comparison_overwrite_cannot_replace_any_input(tmp_path, rf_inputs, output_input):
    source, detected = rf_inputs
    projection = tmp_path / "native.csv"
    projection.write_text("unit_id,-145.0,-135.0\n23,88,132\n7,110,220\n")
    inputs = dict(source=source, projection=projection, detected=detected)
    original = {path: path.read_bytes() for path in inputs.values()}
    with pytest.raises(ValueError, match="output must differ"):
        preparation.prepare_rf_comparison(
            source, inputs[output_input], projection_path=projection,
            detected_rf_path=detected, time_range_s=(0., .2),
            rf_only=True, unit_prefix="paired", overwrite=True,
        )
    assert {path: path.read_bytes() for path in inputs.values()} == original


def test_comparison_rejects_missing_output_that_is_its_projection(tmp_path, rf_inputs):
    source, detected = rf_inputs
    projection = tmp_path / "native" / "missing.csv"
    original = {path: path.read_bytes() for path in rf_inputs}
    with pytest.raises(ValueError, match="output must differ"):
        preparation.prepare_rf_comparison(
            source, projection, projection_path=projection,
            detected_rf_path=detected, time_range_s=(0., .2),
            rf_only=True, unit_prefix="paired",
        )
    assert not projection.exists()
    assert {path: path.read_bytes() for path in rf_inputs} == original


@pytest.mark.parametrize("rf_only", [False, True])
def test_locate_rf_cli_prepares_both_comparisons_from_saved_detections(
    tmp_path, monkeypatch, rf_inputs, rf_only,
):
    source, excitatory = rf_inputs
    inhibitory = tmp_path / "saved_inhibitory.npz"
    masks = np.zeros((3, 3, 2), dtype=np.uint8)
    masks[0, 0, 0] = masks[1, 2, 1] = 1
    save_rf(RFResult(masks, masks, np.array([7, 23, 11]), {}, "supplied"), inhibitory)
    detections = {"excitatory": excitatory, "inhibitory": inhibitory}
    original = {path: path.read_bytes() for path in (source, *detections.values())}
    analyses = {
        kind: dict(
            bin_qc={"unit_ids": np.array([7, 11, 23])},
            summed=SimpleNamespace(n_units=3), units_with_rf=[7, 23],
            units_with_rf_1d=[7, 23], output_paths={"result_2d": path},
        )
        for kind, path in detections.items()
    }

    def analyzed(path, **options):
        assert path == source
        assert options["probe"] == "B"
        assert options["rf_type"] == "both"
        assert options["time_range_s"] == (0., .2)
        return analyses

    monkeypatch.setattr(locate_rf, "analyze_rf_file", analyzed)
    monkeypatch.setattr(_detector, "detect_rf", _unexpected)
    output_dir = tmp_path / "data" / "tc_comparison"
    arguments = [
        "locate_rf.py", str(source), "--probe", "B", "--rf-type", "both",
        "--time-range", "0", ".2", "--unit-prefix", "m21:261006:B",
        "--comparison-output-dir", str(output_dir),
    ]
    if not rf_only:
        arguments.append("--all-rf-rows")
    monkeypatch.setattr(sys, "argv", arguments)
    locate_rf.main()

    suffix = "_rfonly" if rf_only else ""
    for kind in detections:
        native_suffix = "_inhibitory" if rf_only and kind == "inhibitory" else ""
        projection = output_dir.parent / f"source_ProbeB{native_suffix}_1d{suffix}.csv"
        output = output_dir / f"rf_{kind}_x_2d{suffix}_ProbeB.csv"
        assert projection.is_file()
        actual = load_tc(output)
        expected_ids = [23, 7] if rf_only and kind == "excitatory" else [7, 23]
        if rf_only:
            expected = [[88, 132], [110, 220]] if kind == "excitatory" else [[11, 22], [66, 88]]
        else:
            expected = [[154, 286], [308, 572]]
        assert actual.index.tolist() == [f"m21:261006:B:{unit}" for unit in expected_ids]
        np.testing.assert_array_equal(actual.columns, [-145., -135.])
        np.testing.assert_array_equal(actual.to_numpy(), expected)
        np.testing.assert_array_equal(actual.to_numpy(), load_rf_tc(projection).loc[expected_ids].to_numpy())
    assert {path: path.read_bytes() for path in original} == original
