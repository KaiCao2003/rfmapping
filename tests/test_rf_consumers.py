"""RF summaries consume saved results without creating missing detections."""

from pathlib import Path

import numpy as np
import pytest

import hd_rf_fraction
from Utils import rflocate
from Utils.rflocate import RFResult, rf_result_path, save_rf


@pytest.fixture
def saved_results(tmp_path):
    rf_file = tmp_path / "regular.rfmap"
    mask_2d = np.array([
        [[1, 1], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ], dtype=np.uint8)
    center_2d = np.array([
        [[0, 1], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [1, 0]],
    ], dtype=np.uint8)
    result_2d = RFResult(mask_2d, center_2d, np.array([23, 7, 99]), {}, "saved-2d")
    mask_1d = np.array([[[0, 0]], [[1, 0]], [[0, 1]]], dtype=np.uint8)
    result_1d = RFResult(mask_1d, mask_1d, np.array([99, 23, 7]),
                         {"collapse_axis": "x"}, "saved-1d")
    save_rf(result_2d, rf_result_path(rf_file))
    save_rf(result_1d, rf_result_path(rf_file, dimension="1d"))
    return rf_file


@pytest.fixture
def forbid_rf_analysis(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("RF fractions must only read saved detections")

    monkeypatch.setattr(rflocate, "detect_rf", unexpected)
    monkeypatch.setattr(rflocate.detection, "detect_rf", unexpected)
    monkeypatch.setattr(rflocate, "load_rfmap", unexpected)


def test_fraction_reads_unit_ids_from_each_saved_result(saved_results, forbid_rf_analysis):
    assert not saved_results.exists()
    assert hd_rf_fraction.rf_unit_ids(saved_results) == ({23, 7}, {23, 99})


def test_fraction_missing_1d_result_raises_without_detection(saved_results, forbid_rf_analysis):
    missing = rf_result_path(saved_results, dimension="1d")
    missing.unlink()
    with pytest.raises(FileNotFoundError) as exc:
        hd_rf_fraction.rf_unit_ids(saved_results)
    assert Path(exc.value.filename) == missing
    assert not missing.exists()
    assert not saved_results.exists()
