"""Preparation writes only the explicitly requested comparison table."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_comparison_tcs import main
from Utils.rflocate import RFResult, save_rf


@pytest.fixture
def native_csv(tmp_path):
    path = tmp_path / "chosen_y_projection.csv"
    path.write_text(
        "unit_id,-60.0,210.125,0.0\n"
        "23,0,nan,1.2345678901234567\n"
        "7,nan,nan,nan\n"
        "99,2.345678901234567,0,0\n"
        "11,8,9,10\n",
        encoding="utf-8",
    )
    return path


def read_output(path):
    table = pd.read_csv(path, index_col="unit_id", float_precision="round_trip")
    table.columns = table.columns.astype(float)
    return table


def test_rf_export_copies_native_values_without_raw_maps_or_other_outputs(native_csv, tmp_path):
    output = tmp_path / "one_selected_table.csv"
    before = native_csv.read_bytes()
    main(["rf", str(native_csv), str(output), "--unit-prefix", "experiment:day:shank"])

    expected = read_output(native_csv)
    expected.index = expected.index.map(lambda unit: f"experiment:day:shank:{unit}")
    pd.testing.assert_frame_equal(read_output(output), expected)
    assert native_csv.read_bytes() == before
    assert set(tmp_path.iterdir()) == {native_csv, output}


def test_rf_export_unions_saved_detection_ids_in_source_order(native_csv, tmp_path):
    first = tmp_path / "arbitrary_first.npz"
    second = tmp_path / "arbitrary_second.npz"
    for path, ids, detected in (
        (first, [99, 7, 500], [1, 0, 1]),
        (second, [7, 23], [1, 0]),
    ):
        masks = np.asarray(detected, dtype=np.uint8)[:, None, None]
        save_rf(RFResult(masks, masks, np.array(ids), {}, "explicit"), path)
    inputs = {path: path.read_bytes() for path in (native_csv, first, second)}
    output = tmp_path / "selected.csv"
    main(["rf", str(native_csv), str(output), "--select-results", str(first), str(second),
          "--unit-prefix", "matching-units"])

    actual = read_output(output)
    expected = read_output(native_csv).loc[[7, 99]]
    expected.index = pd.Index(["matching-units:7", "matching-units:99"], name="unit_id")
    pd.testing.assert_frame_equal(actual, expected)
    assert {path: path.read_bytes() for path in inputs} == inputs
    assert set(tmp_path.iterdir()) == {*inputs, output}


@pytest.mark.parametrize("hd_class", [None, 3])
def test_hd_export_rebins_rates_and_pairs_explicit_ids(native_csv, tmp_path, hd_class):
    source = tmp_path / "unrelated_name.tc"
    edges = np.linspace(0, 360, 181)
    angles = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    rates = np.array([np.exp(.6 * np.cos(angles)), np.ones(180), np.exp(.6 * np.cos(angles))])
    occupancy = np.linspace(1, 3, 180)
    counts = rates * occupancy
    source.write_text(json.dumps({
        "metadata": {"classification": {}},
        "unit_id": [23, 7, 99],
        "angle_bin_edges_deg": edges.tolist(),
        "occupancy_time_s": occupancy.tolist(),
        "firing_rate_hz": rates.tolist(),
        "spike_counts": counts.tolist(),
        "unit_data": {"rayleigh_p": [.001, .001, .2], "shuffle_p": [.001, .001, .001]},
    }))
    before = source.read_bytes()
    output = tmp_path / "hd.csv"
    arguments = ["hd", str(source), str(output), "--bins", "6", "--unit-prefix", "paired"]
    if hd_class is not None:
        arguments.extend(["--hd-class", str(hd_class)])
    main(arguments)
    rf_output = tmp_path / "rf.csv"
    main(["rf", str(native_csv), str(rf_output), "--unit-prefix", "paired"])

    actual = read_output(output)
    expected_rates = counts.reshape(3, 6, 30).sum(axis=2) / occupancy.reshape(6, 30).sum(axis=1)
    expected_ids = ["paired:23", "paired:7", "paired:99"]
    if hd_class is not None:
        expected_rates = expected_rates[:1]
        expected_ids = expected_ids[:1]
    np.testing.assert_array_equal(actual.columns, [30., 90., 150., -150., -90., -30.])
    np.testing.assert_array_equal(actual.to_numpy(), expected_rates)
    assert actual.index.tolist() == expected_ids
    assert actual.index.isin(read_output(rf_output).index).all()
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {native_csv, source, output, rf_output}


def test_existing_output_skips_until_explicit_overwrite(native_csv, tmp_path):
    output = tmp_path / "selected.csv"
    output.write_text("existing result")
    arguments = ["rf", str(native_csv), str(output), "--unit-prefix", "paired"]
    main(arguments)
    assert output.read_text() == "existing result"
    main([*arguments, "--overwrite"])
    assert read_output(output).index.tolist() == ["paired:23", "paired:7", "paired:99", "paired:11"]


def test_export_cannot_overwrite_an_input(native_csv):
    before = native_csv.read_bytes()
    with pytest.raises(SystemExit):
        main(["rf", str(native_csv), str(native_csv), "--unit-prefix", "paired", "--overwrite"])
    assert native_csv.read_bytes() == before


@pytest.mark.parametrize("arguments", [
    ["hd", "source.tc", "output.csv", "--unit-prefix", "paired"],
    ["rf", "source.csv", "output.csv"],
])
def test_export_requires_explicit_preparation_options(arguments):
    with pytest.raises(SystemExit):
        main(arguments)
