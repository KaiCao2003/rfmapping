"""Execute the notebooks' RF CSV preparation against small recorded-like inputs."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from Utils.direction_comparison import load_rf_profiles, load_tc
from Utils.rflocate import (
    RFMapList, RFResult, load_rf, load_rfmap,
    rf_result_path, save_rf, save_rf_tc,
)
from Utils.rflocate import _detector


NOTEBOOKS = (
    "tc_comparison_pairs.ipynb", "tc_comparison_pairs_inhibitory.ipynb",
    "hd_rf_comparison.ipynb", "hd_rf_ebc_comparison.ipynb",
)


def _csv_blocks(name):
    path = Path(__file__).resolve().parents[1] / name
    notebook = json.loads(path.read_text())
    blocks = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "rf_csv_stem =" not in source:
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
                if "rf_csv_stem" in names:
                    start = index
                if (start is not None
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Name)
                        and (statement.value.func.id == "load_rf_profiles"
                             or (statement.value.func.id == "load_tc"
                                 and any(keyword.arg == "kind" and isinstance(keyword.value, ast.Constant)
                                         and keyword.value.value == "RF" for keyword in statement.value.keywords)))):
                    block = ast.Module(body=node.body[start:index + 1], type_ignores=[])
                    target = ast.unparse(statement.targets[0])
                    blocks.append((compile(block, str(path), "exec"), target))
                    start = None
    return blocks


CASES = [
    (name, index, block, target)
    for name in NOTEBOOKS for index, (block, target) in enumerate(_csv_blocks(name))
]


def test_csv_blocks_cover_all_recordings():
    assert [sum(case[0] == name for case in CASES) for name in NOTEBOOKS] == [5, 5, 1, 1]


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
        load_tc=load_tc, load_rf_profiles=load_rf_profiles,
        save_rf_tc=save_rf_tc, rf_result_path=rf_result_path,
        rf_source=source, rfmap_file=source,
        rf_only=rf_only, rf_detection=rf_type, rf_response_window_s=(0., .2),
        mouse="m14", date=260609, probe="B",
        recording={"mouse": "m14", "date": 260609, "probe": "B"}, profiles={}, rf_name="RF",
    )
    exec(block, namespace)
    suffix = "_inhibitory" if rf_only and rf_type == "inhibitory" else ""
    csv_path = source.parents[4] / f"regular_ProbeB{suffix}_1d{'_rfonly' if rf_only else ''}.csv"
    assert namespace["rf_csv_path"] == csv_path
    table = eval(target, namespace)
    prefix = ("B",) if name == "hd_rf_ebc_comparison.ipynb" else ("m14", "260609", "B")
    index_names = ["probe", "unit_id"] if len(prefix) == 1 else ["mouse", "date", "probe", "unit_id"]
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
