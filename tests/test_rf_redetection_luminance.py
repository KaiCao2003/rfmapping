import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from RF.paths import rf_projection_path, rf_source_path
from RF.rflocate import load_rf_tc
from scripts.redetect_saved_rf_2d import apply_report, main, sha256


def _write_source(path, unit_id):
    counts = np.full((1, 5, 12, 2), 20, dtype=np.uint32)
    counts[0, 1, 2:5] = 48
    counts[0, 3, 7:10] = 4
    metadata = {"unitsSpikeCountsSize": list(counts.shape), "responseUnits": "spike_count",
                "VSTimeWindow": [0., .2]}
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                 unitPool=[unit_id], xPositions=np.arange(12), yPositions=np.arange(5),
                 timeBinEdges=[0., .1, .2], occupancyTimeSec=np.ones((5, 12)),
                 stimulusPresentationCounts=np.ones((5, 12)), **{f"unit_{unit_id}": counts[0]})


def _locate_command(source, comparisons, *, all_rows=False):
    cli = Path(__file__).resolve().parents[1] / "locate_rf.py"
    command = [sys.executable, str(cli), str(source), "--probe", "A", "--drop-bins", "0",
               "--reference-2d", "global", "--sigma-2d", "1", "-1", "--no-plots",
               "--comparison-output-dir", str(comparisons), "--unit-prefix", "m22:261008:A"]
    return [*command, "--all-rf-rows"] if all_rows else command


def test_redetect_same_named_on_off_sources(tmp_path, monkeypatch):
    session = tmp_path / "261008_2"
    data = session / "data"
    comparisons = data / "tc_comparison"
    comparisons.mkdir(parents=True)
    roots = {"rfmapping": 17, "rfmapping_off": 35}
    shared_inventory = []
    inventory = []
    sources = {}
    for root, unit_id in roots.items():
        source = rf_source_path(session, "A", root_kind=root)
        _write_source(source, unit_id)
        sources[root] = source
        native_y = rf_projection_path(source, axis="y")
        native_y.write_text("unit_id,0,1,2,3,4\n")
        shared_inventory.append({"path": str(native_y)})
        inventory.append(dict(path=str(source), probe="ProbeA", root_kind=root, source_kind="regular",
                              rf_root=str(data / root), session_path=str(session), mouse="m22", date="261008",
                              has_saved_2d_detection=True, in_backup=False,
                              session_rf_csvs_shared_or_source_specific=shared_inventory))

    # Both rows see these old ambiguous paths in the same session inventory.
    for ambiguous in (data / "regular_261008_2_ProbeA_1d_rfonly.csv",
                      comparisons / "regular_261008_2_excitatory_x_2d_rfonly_ProbeA.csv"):
        ambiguous.write_text("unowned\n")
        shared_inventory.append({"path": str(ambiguous)})
    # An earlier single-source all-rows export must stay synchronized after redetection.
    subprocess.run(_locate_command(sources["rfmapping"], comparisons, all_rows=True), check=True)
    full_qualified = comparisons / "regular_unitsSpikeCounts_261008_2_excitatory_x_2d_ProbeA.csv"
    full_alias = comparisons / "rf_excitatory_x_2d_ProbeA.csv"
    expected_full = full_qualified.read_bytes()
    full_qualified.write_text("stale qualified file\n")
    full_alias.write_text("stale alias\n")
    shared_inventory.extend({"path": str(path)} for path in (full_qualified, full_alias))
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"sources": inventory}))
    raw_hashes = {source: sha256(source) for source in sources.values()}
    report_path = tmp_path / "report/run.json"
    args = ["redetect", "--inventory", str(inventory_path), "--report", str(report_path),
            "--sigma-ext", "1.0", "--sigma-inh", "-1.0", "--drop-ext", "0", "--drop-inh", "0"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    report = json.loads(report_path.read_text())
    first, second = report["sources"]
    targets = [{Path(entry["target"]) for entry in row["replacements"]} for row in (first, second)]
    assert targets[0].isdisjoint(targets[1])
    assert all(not rf_projection_path(source).exists() for source in sources.values())

    # A previously staged report sharing a destination must abort before its first write.
    all_targets = targets[0] | targets[1]
    before_apply = {path: sha256(path) for path in all_targets if path.exists()}
    second["replacements"].append(first["replacements"][0])
    with pytest.raises(ValueError, match="share target"):
        apply_report(report, report_path)
    assert {path: sha256(path) for path in all_targets if path.exists()} == before_apply
    monkeypatch.setattr(sys, "argv", [*args, "--apply"])
    main()
    for root, unit_id in roots.items():
        source = sources[root]
        assert sha256(source) == raw_hashes[source]
        np.testing.assert_array_equal(load_rf_tc(rf_projection_path(source, axis="y")).index, [unit_id])
        suffix = "" if root == "rfmapping" else "_off"
        for polarity, label in (("excitatory", "ext"), ("inhibitory", "inh")):
            native = load_rf_tc(rf_projection_path(source, rf_type=polarity))
            assert native.index.tolist() == [unit_id]
            name = f"regular_unitsSpikeCounts_261008_2{suffix}_{polarity}_x_2d_rfonly_ProbeA.csv"
            comparison = load_rf_tc(comparisons / name)
            assert comparison.index.tolist() == [f"m22:261008:A:{unit_id}"]
            np.testing.assert_array_equal(comparison.values, native.values)
            pooled = source.parents[2] / "all/regular" / f"unitList_2d_{label}.npy"
            np.testing.assert_array_equal(np.load(pooled), [["A", str(unit_id)]])
            if root == "rfmapping":
                alias = comparisons / f"rf_{polarity}_x_2d_rfonly_ProbeA.csv"
                assert alias.read_bytes() == (comparisons / name).read_bytes()
    assert (data / "regular_261008_2_ProbeA_1d_rfonly.csv").read_text() == "unowned\n"
    assert (comparisons / "regular_261008_2_excitatory_x_2d_rfonly_ProbeA.csv").read_text() == "unowned\n"
    assert full_qualified.read_bytes() == full_alias.read_bytes() == expected_full


@pytest.mark.parametrize("all_rows", [False, True])
def test_locate_cli_separates_organized_sources(tmp_path, all_rows):
    session = tmp_path / "261008_2"
    comparisons = tmp_path / "comparisons"
    suffix = "" if all_rows else "_rfonly"
    recordings = [
        ("rfmapping", "regular", 17, "rf"),
        ("rfmapping_off", "regular", 35, "regular_unitsSpikeCounts_261008_2_off"),
        ("rfmapping", "rotation_30", 49, "rotation_30_unitsSpikeCounts_261008_2"),
    ]
    for root, kind, unit_id, _ in recordings:
        source = rf_source_path(session, "A", root_kind=root, kind=kind)
        _write_source(source, unit_id)
        command = _locate_command(source, comparisons, all_rows=all_rows)
        regular_on = root == "rfmapping" and kind == "regular"
        if regular_on:
            comparisons.mkdir()
            prefix = "regular_unitsSpikeCounts_261008_2" if all_rows else "rf"
            conflict = comparisons / f"{prefix}_inhibitory_x_2d{suffix}_ProbeA.csv"
            conflict.write_text("pre-existing comparison\n")
            failed = subprocess.run(command, capture_output=True, text=True)
            assert failed.returncode and f"{conflict} already exists" in failed.stderr
            assert list(comparisons.iterdir()) == [conflict]
            assert not list(source.parent.glob("tc_*.csv"))
            conflict.unlink()
        subprocess.run(command, check=True)
        if regular_on:
            for path in comparisons.glob("*.csv"):
                path.write_text(f"old {path.name}\n")
            subprocess.run([*command, "--overwrite-comparison"], check=True)

    for root, kind, unit_id, prefix in recordings:
        source = rf_source_path(session, "A", root_kind=root, kind=kind)
        for polarity, label in (("excitatory", "ext"), ("inhibitory", "inh")):
            projection_name = "tc_x_all.csv" if all_rows else f"tc_x_rfonly_{label}.csv"
            native = load_rf_tc(source.parent / projection_name)
            assert native.index.tolist() == [unit_id]
            filename = f"{polarity}_x_2d{suffix}_ProbeA.csv"
            comparison = comparisons / f"{prefix}_{filename}"
            table = load_rf_tc(comparison)
            assert table.index.tolist() == [f"m22:261008:A:{unit_id}"]
            np.testing.assert_array_equal(table.values, native.values)
            if prefix == "rf":
                qualified = comparisons / f"regular_unitsSpikeCounts_261008_2_{filename}"
                assert comparison.read_bytes() == qualified.read_bytes()
            if all_rows:
                np.testing.assert_array_equal(native.iloc[0], [200, 200, 256, 256, 256, 200,
                                                              200, 168, 168, 168, 200, 200])
