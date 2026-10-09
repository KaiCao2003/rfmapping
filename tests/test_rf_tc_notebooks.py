"""Execute notebook RF CSV reads and preparation against small recorded-like inputs."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Utils import direction_comparison as comparison
from Utils import rflocate
from Utils import tc_comparison as paired_comparison
from Utils import tc_preparation as preparation
from Utils.direction_comparison import load_rf_profiles, tc_loader
from Utils.rflocate import (
    RFMapList, RFResult, load_rf, load_rfmap,
    rf_result_path, save_rf, save_rf_tc,
)
from Utils.rflocate import _detector


PAIRED_NOTEBOOKS = ("tc_comparison_pairs.ipynb", "tc_comparison_pairs_inhibitory.ipynb")
GENERATION_NOTEBOOKS = ("hd_rf_comparison.ipynb",)


def _csv_blocks(name, start_name="rf_csv_stem"):
    path = Path(__file__).resolve().parents[1] / name
    notebook = json.loads(path.read_text())
    blocks = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or f"{start_name} =" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.For)):
                continue
            start = None
            for index, statement in enumerate(node.body):
                if not isinstance(statement, ast.Assign):
                    continue
                names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
                if start_name in names:
                    start = index
                if (start is not None
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and (statement.value.func.id == "load_rf_profiles"
                             or (statement.value.func.id == "tc_loader"
                                 and any(keyword.arg == "kind" and isinstance(keyword.value, ast.Constant)
                                         and keyword.value.value == "RF" for keyword in statement.value.keywords)))):
                    block = ast.Module(body=node.body[start:index + 1], type_ignores=[])
                    target = ast.unparse(statement.targets[0])
                    blocks.append((compile(block, str(path), "exec"), target))
                    start = None
    return blocks


CASES = [
    (name, index, block, target)
    for name in GENERATION_NOTEBOOKS for index, (block, target) in enumerate(_csv_blocks(name))
]


def test_csv_blocks_cover_all_recordings():
    assert [sum(case[0] == name for case in CASES) for name in GENERATION_NOTEBOOKS] == [1]


def _paired_cells(name):
    path = Path(__file__).resolve().parents[1] / name
    notebook = json.loads(path.read_text())
    config, prepared = None, []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "".join(line for line in source.splitlines(keepends=True) if not line.startswith("%"))
        tree = ast.parse(source)
        assigned = {target.id for statement in tree.body if isinstance(statement, ast.Assign)
                    for target in statement.targets if isinstance(target, ast.Name)}
        if "root" in assigned:
            config = tree
        if any(target.startswith("hd_normalized_") for target in assigned):
            last_preparation = max(
                index for index, statement in enumerate(tree.body)
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name)
                    and target.id.startswith(("hd_normalized_", "rf_normalized_"))
                    for target in statement.targets
                )
            )
            prepared.append(ast.Module(body=tree.body[:last_preparation + 1], type_ignores=[]))
    assert config is not None
    assert prepared
    return path, config, prepared


@pytest.mark.parametrize("name", PAIRED_NOTEBOOKS)
def test_paired_notebooks_prepare_explicit_sources_then_load_csv(name):
    path = Path(__file__).resolve().parents[1] / name
    notebook = json.loads(path.read_text())
    calls = set()
    counts = {function: 0 for function in (
        "tc_loader", "prepare_hd_tc", "prepare_rf_comparison", "prep_hd_tc", "prep_rf_tc",
    )}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "".join(line for line in source.splitlines(keepends=True) if not line.startswith("%"))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                    if node.func.id in counts:
                        counts[node.func.id] += 1
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    if name == "tc_comparison_pairs.ipynb":
        assert counts["prep_hd_tc"] == counts["prep_rf_tc"] > 0
        assert counts["tc_loader"] == counts["prepare_hd_tc"] == counts["prepare_rf_comparison"] == 0
    else:
        assert counts["prepare_hd_tc"] == counts["prepare_rf_comparison"] > 0
        assert counts["tc_loader"] == counts["prepare_hd_tc"] + counts["prepare_rf_comparison"]
    assert not calls.intersection({
        "RFMapList", "load_rfmap", "load_rf", "load_hd_profiles", "save_rf_tc", "save_tc",
        "sum_to_1d", "detect_rf", "hd_pick", "select_rf_profiles", "update_hd_classification",
    })


def _seed_paired_comparison_csvs(prepared, namespace, monkeypatch):
    """Seed the paths requested by the notebook and its actual preparation helpers."""
    specs = []

    def collect(kind):
        def record(source, output, **options):
            specs.append((kind, source, output, options))
            prefix = options["unit_prefix"]
            values = np.tile(np.arange(1., 31.), (4, 1))
            values[0, 5] = np.nan
            unit_ids = [23, 7, 99, 42] if kind == "HD" else [7, 23, 99, 11]
            if kind == "RF":
                values[0, 2:4] = [np.nan, 0.]
                values[1, :2] = 0.
                values[2, :3] = 0.
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(values, columns=np.arange(-174., 180., 12.),
                         index=pd.Index([f"{prefix}:{unit}" for unit in unit_ids], name="unit_id")).to_csv(output)
            return output
        return record

    scope = {**namespace, "prepare_hd_tc": collect("HD"), "prepare_rf_comparison": collect("RF")}
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "prepare_hd_tc", scope["prepare_hd_tc"])
        patch.setattr(preparation, "prepare_rf_comparison", scope["prepare_rf_comparison"])
        for tree in prepared:
            exec(compile(tree, "comparison cache fixtures", "exec"), scope)
    return specs


@pytest.mark.parametrize("rf_only", [False, True])
@pytest.mark.parametrize("name", PAIRED_NOTEBOOKS)
def test_paired_comparison_reuses_existing_csvs_for_all_declared_sections(
    tmp_path, monkeypatch, name, rf_only,
):
    path, config, prepared = _paired_cells(name)
    overrides = dict(root=tmp_path, rf_only=rf_only)
    for statement in config.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
            if isinstance(target, ast.Name) and target.id in overrides:
                statement.value = ast.parse(f"_config[{target.id!r}]", mode="eval").body
    namespace = {"_config": overrides}
    exec(compile(ast.fix_missing_locations(config), str(path), "exec"), namespace)
    namespace.update(prep_hd_tc=paired_comparison.prep_hd_tc, prep_rf_tc=paired_comparison.prep_rf_tc)
    specs = _seed_paired_comparison_csvs(prepared, namespace, monkeypatch)
    prefixes = {options["unit_prefix"] for _, _, _, options in specs}
    if name == "tc_comparison_pairs.ipynb":
        assert "m21:261006:A" in prefixes
    assert {(kind, options["unit_prefix"]) for kind, _, _, options in specs} == {
        (kind, prefix) for prefix in prefixes for kind in ("HD", "RF")
    }
    saved = {}
    for kind, source, output, options in specs:
        if kind == "RF":
            assert options["rf_only"] is rf_only
            assert options["time_range_s"] == (0., .2)
        saved[output] = output.read_bytes()
        assert not source.exists()

    def unexpected(*args, **kwargs):
        pytest.fail("Existing comparison CSVs must bypass raw sources and classification/detection")

    for function in ("load_hd_profiles", "load_rfmap", "load_rf", "load_rf_tc"):
        monkeypatch.setattr(preparation, function, unexpected)
    for function in ("load_hd_profiles", "load_ebc_profiles", "load_rfmap", "load_detected_rf",
                     "read_formatted_json", "update_hd_classification", "hd_pick", "select_rf_profiles", "save_tc"):
        monkeypatch.setattr(comparison, function, unexpected)
    for function in ("load_rfmap", "load_rf", "save_rf_tc"):
        monkeypatch.setattr(rflocate, function, unexpected)
    monkeypatch.setattr(_detector, "detect_rf", unexpected)
    for tree in prepared:
        exec(compile(tree, str(path), "exec"), namespace)

    normalized_names = [key for key in namespace if key.startswith("hd_normalized_")]
    if name == "tc_comparison_pairs.ipynb":
        assert "hd_normalized_m21" in normalized_names
    for hd_name in normalized_names:
        key = hd_name.removeprefix("hd_normalized_")
        hd, rf = namespace[hd_name], namespace[f"rf_normalized_{key}"]
        raw_hd, raw_rf = namespace[f"fm_{key}"], namespace[f"rf_{key}"]
        unit_keys = raw_hd.index.intersection(raw_rf.index).tolist()
        assert hd.index.names == rf.index.names == ["unit_id"]
        assert hd.index.tolist() == rf.index.tolist() == unit_keys
        assert unit_keys
        assert [unit.rsplit(":", 1)[-1] for unit in unit_keys] == ["23", "7"] * (len(unit_keys) // 2)
        assert {unit.rsplit(":", 1)[-1] for unit in raw_hd.index} == {"23", "7", "99", "42"}
        assert {unit.rsplit(":", 1)[-1] for unit in raw_rf.index} == {"7", "23", "11"}
        for table, raw in ((hd, raw_hd), (rf, raw_rf)):
            values = raw.loc[unit_keys].to_numpy()
            np.testing.assert_allclose(table.to_numpy(), values / np.nanmax(values, axis=1)[:, None], equal_nan=True)
        if name == "tc_comparison_pairs.ipynb":
            assert raw_hd.attrs["response_units"] == raw_rf.attrs["response_units"] == "normalized"
        else:
            assert raw_hd.attrs["response_units"] == "Hz"
            assert raw_rf.attrs["response_units"] == "spike_count"
    assert set(tmp_path.rglob("*.*")) == set(saved)
    assert all(file.read_bytes() == contents for file, contents in saved.items())


@pytest.mark.parametrize("rf_only", [False, True])
@pytest.mark.parametrize("name,index,block,target", CASES, ids=[f"{name}-{index}" for name, index, _, _ in CASES])
def test_comparison_csv_generation_and_existing_file_read(tmp_path, monkeypatch, name, index, block, target, rf_only):
    source = tmp_path / "m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeB/regular.rfmap"
    source.parent.mkdir(parents=True)
    grid = np.array([[1, 2], [10, 20], [3, 4]])
    responses = np.stack([grid, np.zeros_like(grid), grid * 2])
    counts = np.stack([responses, responses * 10], axis=-1)
    source.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7, 11, 23], "xPositions": [-145., -135.], "yPositions": [-20., 0., 20.],
        "timeBinEdges": [0., .1, .2],
    }))
    masks = np.array([[[1, 0], [0, 0], [0, 1]], [[0, 0], [1, 0], [0, 0]]], dtype=np.uint8)
    centers = np.zeros_like(masks)
    centers[0, 0, 0] = centers[1, 1, 0] = 1
    rf_type = "inhibitory" if "inhibitory" in name else "excitatory"
    detected_path = rf_result_path(source, rf_type=rf_type)
    save_rf(RFResult(masks, centers, np.array([23, 7]), {}, "supplied"), detected_path)

    def unexpected(*args, **kwargs):
        pytest.fail("A saved TC read must not regenerate or detect")

    monkeypatch.setattr(_detector, "detect_rf", unexpected)
    namespace = dict(
        RFMapList=RFMapList, load_rfmap=load_rfmap, load_rf=load_rf,
        tc_loader=tc_loader, load_rf_profiles=load_rf_profiles,
        rf_range_overrides={"m14": [-150, 150]}, tcRange=comparison.tcRange,
        save_rf_tc=save_rf_tc, rf_result_path=rf_result_path,
        rf_source=source, rfmap_file=source,
        rf_only=rf_only, rf_detection=rf_type, rf_response_window_s=(0., .2),
        mouse="m14", date=260609, probe="B",
        recording={"mouse": "m14", "date": 260609, "probe": "B"}, profiles={}, rf_name="RF",
        prepare_rf_projection=preparation.prepare_rf_projection,
        prepare_rf_comparison=preparation.prepare_rf_comparison, unit_prefix="m14:260609:B",
    )
    exec(block, namespace)
    suffix = "_inhibitory" if rf_only and rf_type == "inhibitory" else ""
    csv_path = source.parents[4] / f"regular_ProbeB{suffix}_1d{'_rfonly' if rf_only else ''}.csv"
    assert namespace["rf_csv_path"] == csv_path
    table = eval(target, namespace)
    prefix = ("B",)
    index_names = ["probe", "unit_id"]
    assert table.index.names == index_names
    assert table.index.tolist() == [(*prefix, unit) for unit in ([23, 7] if rf_only else [7, 11, 23])]
    expected = [[88, 132], [110, 220]] if rf_only else [[154, 286], [0, 0], [308, 572]]
    np.testing.assert_array_equal(table.to_numpy(), expected)
    np.testing.assert_array_equal(table.columns, [-145., -135.])

    # The existing-file branch must use the CSV even without either source input.
    source.unlink()
    detected_path.unlink()
    csv_path.write_text("unit_id,-145.0,-135.0\n123,7,9\n")
    before = csv_path.read_bytes()
    namespace.update(load_rfmap=unexpected, load_rf=unexpected, save_rf_tc=unexpected)
    exec(block, namespace)
    table = eval(target, namespace)
    assert table.index.tolist() == [(*prefix, 123)]
    np.testing.assert_array_equal(table.to_numpy(), [[7, 9]])
    assert csv_path.read_bytes() == before
