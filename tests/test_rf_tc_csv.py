"""RF TC CSVs preserve prepared values and never generate analysis inputs."""

import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from Utils.rflocate import RFMapList, asrfmap, load_rf_tc, save_rf_tc
from Utils.rflocate import detection, io


def _prepared_maps(axis):
    coordinates = np.array([-170., -12.5, 190.12345678912345])
    values = np.array([
        [0., np.nan, 1.2345678901234567],
        [2.345678901234567, 0., 0.0000001234567891234567],
        [np.nan, np.nan, np.nan],
    ])
    maps = []
    for index, (unit_id, row) in enumerate(zip([23, 7, 99], values, strict=True)):
        matrix = row[None, :] if axis == "x" else row[:, None]
        maps.append(replace(
            asrfmap(matrix), unit_id=unit_id, unit_index=index,
            **{f"{axis}_positions": coordinates},
        ))
    return RFMapList(maps, "<array>"), coordinates, values


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rf_tc_round_trip_preserves_native_coordinates_values_and_unit_order(tmp_path, axis):
    maps, coordinates, values = _prepared_maps(axis)
    path = tmp_path / "chosen-profile.csv"
    save_rf_tc(maps, path, axis=axis)
    loaded = load_rf_tc(path)
    np.testing.assert_array_equal(loaded.index, [23, 7, 99])
    assert loaded.index.name == "unit_id"
    np.testing.assert_array_equal(loaded.columns, coordinates)
    np.testing.assert_array_equal(loaded.to_numpy(), values)
    assert set(tmp_path.iterdir()) == {path}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 4
    assert rows[0] == ["unit_id", *map(str, coordinates)]
    assert all(len(row) == 4 for row in rows)


@pytest.mark.parametrize("values, error", [
    (np.ones((2, 3)), "singleton spatial axis"),
    (np.ones((1, 3, 2)), "exactly one time bin"),
])
def test_rf_tc_save_requires_prepared_data_without_implicit_summation(tmp_path, values, error):
    path = tmp_path / "profile.csv"
    maps = RFMapList([asrfmap(values)], "<array>")
    with pytest.raises(ValueError, match=error):
        save_rf_tc(maps, path)
    assert not path.exists()


def test_rf_tc_save_rejects_different_native_coordinate_axes(tmp_path):
    maps, _, _ = _prepared_maps("x")
    different = replace(maps[1], x_positions=np.array([-170., -11., 190.]))
    mismatched = RFMapList([maps[0], different], "<array>")
    with pytest.raises(ValueError, match="share one coordinate axis"):
        save_rf_tc(mismatched, tmp_path / "profile.csv")


def test_missing_rf_tc_only_attempts_the_requested_file(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Loading an RF TC must not generate or analyze data")

    monkeypatch.setattr(detection, "detect_rf", unexpected)
    monkeypatch.setattr(io, "load_rfmap", unexpected)
    monkeypatch.setattr(io, "load_rf", unexpected)
    monkeypatch.setattr(io, "save_rf_tc", unexpected)
    path = tmp_path / "exact-name.csv"
    with pytest.raises(FileNotFoundError) as exc:
        load_rf_tc(path)
    assert Path(exc.value.filename) == path
    assert list(tmp_path.iterdir()) == []


def test_rf_tc_exports_do_not_import_pandas_with_core_package():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; from Utils.rflocate import save_rf_tc, load_rf_tc; "
         "assert 'pandas' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
