from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RF.rflocate import RFMapList, RFResult, asrfmap, load_rf_tc, rf_result_path, save_rf, save_rf_tc
from scripts.rf_redetection_csvs import stage_existing_csvs
from Utils.direction_comparison import save_tc, tc_loader


def _inputs(tmp_path, *, root_kind="rfmapping", name="regular_unitsSpikeCounts_260101_2"):
    data = tmp_path / "260101_2/data"
    source = data / root_kind / "good/-100_400_1ms/ProbeA" / f"{name}.rfmap"
    source.parent.mkdir(parents=True)
    first = asrfmap(np.arange(1, 9).reshape(2, 4), end_time=.2)
    counts = RFMapList([
        replace(first, unit_index=i, unit_id=uid, source_path=source)
        for i, uid in enumerate((7, 11, 23))
    ], source)
    mask = np.zeros((3, 2, 4), dtype=np.uint8)
    mask[0, 0, 1] = 1
    result = RFResult(mask_2d=mask, center_2d=mask, unit_ids=[7, 11, 23], manifest={}, cache_key="new-2d")
    mask_1d = np.zeros((3, 1, 4), dtype=np.uint8)
    mask_1d[1, 0, 2] = 1
    saved = RFResult(mask_2d=mask_1d, center_2d=mask_1d, unit_ids=[7, 11, 23], manifest={}, cache_key="saved-1d")
    save_rf(saved, rf_result_path(source, dimension="1d"))
    item = dict(path=str(source), probe="ProbeA", root_kind=root_kind, source_kind="regular",
                session_path=str(data.parent), mouse="m", date="260101",
                session_rf_csvs_shared_or_source_specific=[])
    return item, counts, result, data


def _existing(item, path, *, axis="x", prefixed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = ["m:260101:A:7", "m:260101:A:11"] if prefixed else [7, 11]
    table = pd.DataFrame(99., index=pd.Index(ids, name="unit_id"), columns=np.arange(4 if axis == "x" else 2))
    (save_tc if prefixed else save_rf_tc)(table, path)
    item["session_rf_csvs_shared_or_source_specific"].append({"path": str(path)})
    return path


def test_refresh_preserves_formats_and_1d_only_exports(tmp_path):
    item, counts, result, data = _inputs(tmp_path)
    files = {}
    for selection in ("1d", "2d", "either", "all"):
        for only in (False, True):
            name = f"rf_excitatory_x_{selection}{'_rfonly' if only else ''}_ProbeA.csv"
            files[(selection, only)] = _existing(item, data / name)
    comparison = _existing(item, data / "tc_comparison/rf_excitatory_x_2d_rfonly_ProbeA.csv", prefixed=True)
    y_path = _existing(item, data / f"{Path(item['path']).stem}_ProbeA_1d_y_rfonly.csv", axis="y")
    before = {Path(f["path"]): Path(f["path"]).read_bytes() for f in item["session_rf_csvs_shared_or_source_specific"]}

    replacements = stage_existing_csvs(item, counts, result, "excitatory", tmp_path / "staged")
    staged = {Path(row["target"]): Path(row["staged"]) for row in replacements}
    assert files[("1d", False)] not in staged
    assert files[("all", False)] not in staged
    assert len(staged) == 8
    assert {path: path.read_bytes() for path in before} == before
    selected = load_rf_tc(staged[files[("2d", True)]])
    assert selected.index.tolist() == [7]
    np.testing.assert_array_equal(selected.to_numpy(), [[1, 2, 3, 4]])
    independent = load_rf_tc(staged[files[("1d", True)]])
    assert independent.index.tolist() == [11]
    assert independent.isna().all().all()
    either = load_rf_tc(staged[files[("either", False)]])
    assert either.index.tolist() == [7, 11]
    np.testing.assert_array_equal(either.to_numpy(), [[6, 8, 10, 12], [6, 8, 10, 12]])
    prefixed = tc_loader(staged[comparison])
    assert prefixed.index.tolist() == ["m:260101:A:7"]
    np.testing.assert_array_equal(prefixed.to_numpy(), selected.to_numpy())
    vertical = load_rf_tc(staged[y_path])
    np.testing.assert_array_equal(vertical.loc[7].to_numpy(), [2, 6])
    assert vertical.loc[11].isna().all()


@pytest.mark.parametrize("root_kind,name", [
    ("rfmapping_off", "regular_unitsSpikeCounts_260101_2_off"),
    ("rfmapping", "rotation_30_unitsSpikeCounts_260101_2"),
    ("rfmapping", "regular_unitsSpikeCounts_260101_2_vertical_bar_pooled_bin3deg"),
])
def test_other_stimuli_do_not_replace_regular_on_aliases(tmp_path, root_kind, name):
    item, counts, result, data = _inputs(tmp_path, root_kind=root_kind, name=name)
    alias = _existing(item, data / "rf_excitatory_x_2d_rfonly_ProbeA.csv")
    native = _existing(item, data / f"{name}_ProbeA_1d_rfonly.csv")
    rows = stage_existing_csvs(item, counts, result, "excitatory", tmp_path / "staged")
    assert [Path(row["target"]) for row in rows] == [native]
    assert alias.read_text().splitlines()[1].startswith("7,99.0")
