"""RF reading, numerical calculation, and persistence have explicit boundaries."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RF.rflocate import RFMapList, asrfmap, load_rfmap, load_rf_tc, rf_tc_table, save_rf_tc
from RF.rflocate import workflow


def _loaded_maps():
    values = np.full((3, 12), 4.)
    values[1, 2:5] = [15, 18, 15]
    values[1, 8:11] = [.5, .25, .5]
    first = replace(asrfmap(values, end_time=.2), unit_id=23,
                    presentation_counts=np.ones(values.shape))
    second = replace(first, unit_id=7, unit_index=1, spike_counts=first.spike_counts * 2)
    return RFMapList([first, second], "<array>")


def _unexpected_io(*args, **kwargs):
    raise AssertionError("this numerical operation must not read, reconstruct, or save files")


def test_analysis_entrypoints_require_drop_bins_before_reading(tmp_path, monkeypatch):
    maps = _loaded_maps()
    monkeypatch.setattr(workflow, "load_rfmap", _unexpected_io)
    with pytest.raises(TypeError, match="drop_bins"):
        workflow.analyze_rf_maps(maps, probe="A", reference_2d="global")
    with pytest.raises(TypeError, match="drop_bins"):
        workflow.analyze_rf_file(tmp_path / "missing.rfmap", probe="A", reference_2d="global")


def test_analysis_entrypoints_require_reference_2d_before_reading(tmp_path, monkeypatch):
    maps = _loaded_maps()
    monkeypatch.setattr(workflow, "load_rfmap", _unexpected_io)
    with pytest.raises(TypeError, match="reference_2d"):
        workflow.analyze_rf_maps(maps, probe="A", drop_bins=0)
    with pytest.raises(TypeError, match="reference_2d"):
        workflow.analyze_rf_file(tmp_path / "missing.rfmap", probe="A", drop_bins=0)


def test_rf_tc_conversion_preserves_native_data_and_saver_only_accepts_tables(tmp_path, monkeypatch):
    source = replace(asrfmap([[0, 2, np.nan], [4, 5, 6]]),
                     x_positions=np.array([-15., -5., 5.]))
    maps = RFMapList([replace(source, unit_id=23)], "<array>").sum_to_1d()
    with monkeypatch.context() as context:
        context.setattr(Path, "open", _unexpected_io)
        table = rf_tc_table(maps)
    np.testing.assert_array_equal(table.index, [23])
    np.testing.assert_array_equal(table.columns, [-15, -5, 5])
    np.testing.assert_allclose(table.to_numpy(), [[4, 7, np.nan]], equal_nan=True)
    assert table.attrs["response_units"] == "spike_count"
    before = table.copy(deep=True)
    output = tmp_path / "native.csv"
    monkeypatch.setattr(RFMapList, "to_1d_array", _unexpected_io)
    save_rf_tc(table, output)
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(load_rf_tc(output), before, check_dtype=False)
    rejected = tmp_path / "not_generated.csv"
    with pytest.raises(TypeError, match="rf_tc_table"):
        save_rf_tc(maps, rejected)
    assert not rejected.exists()


def test_analyze_loaded_maps_computes_both_polarities_without_io(monkeypatch):
    maps = _loaded_maps()
    before = maps.to_4d_array().copy()
    monkeypatch.setattr(workflow, "load_rfmap", _unexpected_io)
    monkeypatch.setattr(workflow, "save_rf_analysis", _unexpected_io)
    monkeypatch.setattr("RF.rflocate.rates.resolve_presentation_counts", _unexpected_io)
    monkeypatch.setattr(np, "load", _unexpected_io)
    monkeypatch.setattr(np, "save", _unexpected_io)
    result = workflow.analyze_rf_maps(maps, probe="A", rf_type="both", drop_bins=0, reference_2d="global")
    assert set(result) == {"excitatory", "inhibitory"}
    for polarity, analysis in result.items():
        assert analysis["result_2d"].unit_ids.tolist() == [23, 7]
        assert analysis["result_2d"].mask_2d.shape == (2, 3, 12)
        assert analysis["result_1d"].mask_2d.shape == (2, 1, 12)
        assert (analysis["parameters"]["sigma_2d"] > 0) == (polarity == "excitatory")
        assert analysis["result_2d"].mask_2d.any()
        assert analysis["result_1d"].mask_2d.any()
    np.testing.assert_array_equal(maps.to_4d_array(), before)


def test_analyze_loaded_maps_never_reconstructs_missing_exposure(monkeypatch):
    maps = _loaded_maps()
    maps = RFMapList([replace(item, presentation_counts=None) for item in maps], maps.source_path)
    monkeypatch.setattr("RF.rflocate.rates.resolve_presentation_counts", _unexpected_io)
    with pytest.raises(ValueError, match="presentation_counts"):
        workflow.analyze_rf_maps(maps, probe="A", drop_bins=2, reference_2d="global")


def test_explicit_file_workflow_matches_loaded_calculation(tmp_path):
    maps = _loaded_maps()
    counts = maps.to_4d_array()
    path = tmp_path / "regular.rfmap"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": maps.unit_ids, "xPositions": maps[0].x_positions.tolist(),
        "yPositions": maps[0].y_positions.tolist(), "timeBinEdges": [0, .2],
        "stimulusPresentationCounts": maps[0].presentation_counts.tolist(),
    }))
    before = path.read_bytes()
    expected = workflow.analyze_rf_maps(load_rfmap(path), probe="A", rf_type="both", drop_bins=0, reference_2d="global")
    actual = workflow.analyze_rf_file(path, probe="A", rf_type="both", drop_bins=0, save_results=False, reference_2d="global")
    assert list(tmp_path.iterdir()) == [path]
    for polarity in expected:
        for name in ("result_2d", "result_1d"):
            first, second = expected[polarity][name], actual[polarity][name]
            np.testing.assert_array_equal(first.mask_2d, second.mask_2d)
            np.testing.assert_array_equal(first.center_2d, second.center_2d)
            assert first.cache_key == second.cache_key
        workflow.save_rf_analysis(actual[polarity])
        assert all(output.exists() for output in actual[polarity]["output_paths"].values())
    assert path.read_bytes() == before
