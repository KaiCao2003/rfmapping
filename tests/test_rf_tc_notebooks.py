"""Execute ordinary notebook read cells against saved, recorded-like CSVs."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from HD import profiles as hd_profiles
from EBB.ebc_geometry import BASLER_SIZE_CM
from RF import profiles as rf_profiles
from RF.rflocate import rf_result_path
from Utils import direction_comparison as comparison


PAIRED_NOTEBOOKS = ("tc_comparison_pairs.ipynb", "tc_comparison_pairs_inhibitory.ipynb")


def _trees(name):
    path = Path(__file__).resolve().parents[1] / name
    cells = json.loads(path.read_text())["cells"]
    trees = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "".join(line for line in source.splitlines(keepends=True) if not line.startswith(("%", "!")))
        trees.append(ast.parse(source, str(path)))
    return path, trees


def _assigned_names(statement):
    if isinstance(statement, ast.Assign):
        return {node.id for target in statement.targets for node in ast.walk(target)
                if isinstance(node, ast.Name)}
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return {statement.target.id}
    return set()


def _paired_namespace(name, root, rf_only):
    path, trees = _trees(name)
    config = next(tree for tree in trees if any("root" in _assigned_names(node) for node in tree.body))
    for node in config.body:
        names = _assigned_names(node)
        if names.intersection({"root", "rf_only"}):
            key = next(iter(names))
            node.value = ast.parse(f"_config[{key!r}]", mode="eval").body
    namespace = dict(
        _config=dict(root=root, rf_only=rf_only), Path=Path,
        load_hd_comparison=hd_profiles.load_hd_comparison, prep_hd_tc=hd_profiles.prep_hd_tc,
        load_rf_comparison=rf_profiles.load_rf_comparison, prep_rf_tc=rf_profiles.prep_rf_tc,
        tc_loader=comparison.tc_loader, tcRange=comparison.tcRange,
        concat=comparison.concat, resample_profiles=comparison.resample_profiles,
        normalize_tc=comparison.normalize_tc, minmax_profiles=comparison.minmax_profiles,
        rf_pick=rf_profiles.rf_pick,
    )
    exec(compile(ast.fix_missing_locations(config), str(path), "exec"), namespace)
    readers = [node for tree in trees for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "require_saved_input"]
    exec(compile(ast.Module(body=readers, type_ignores=[]), str(path), "exec"), namespace)
    prepared = []
    for tree in trees:
        indices = [i for i, node in enumerate(tree.body)
                   if any(name.startswith(("hd_normalized_", "rf_normalized_"))
                          for name in _assigned_names(node))]
        if indices:
            prepared.append(ast.Module(body=tree.body[:max(indices) + 1], type_ignores=[]))
    assert prepared
    return path, namespace, prepared


def _saved_input(path):
    """Fixture tables retain complete IDs, missing bins, measured zeros, and constants."""
    kind = "HD" if path.name.startswith("hd_") else "RF"
    probe = path.stem.rsplit("Probe", 1)[1]
    prefix = f"{path.parents[4].name}:{path.parents[3].name}:{probe}"
    ids = [23, 7, 99, 42] if kind == "HD" else [7, 23, 99, 11]
    values = np.tile(np.arange(1., 31.), (4, 1))
    if kind == "RF":
        values[0] = 5.
        values[0, 5] = np.nan
        values[1, :2] = 0.
        values[2, :3] = 0.
    else:
        values[0, 5] = np.nan
    table = pd.DataFrame(values, columns=np.arange(-174., 180., 12.),
                         index=pd.Index([f"{prefix}:{unit}" for unit in ids], name="unit_id"))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)


def _run(path, prepared, namespace):
    for tree in prepared:
        exec(compile(tree, str(path), "exec"), namespace)


def _fixture_csvs(path, prepared, namespace, monkeypatch):
    # Discover actual notebook paths through its loaders, then use untouched readers.
    def fixture_load(source, **options):
        _saved_input(source)
        return comparison.tc_loader(source, **options)

    with monkeypatch.context() as patch:
        patch.setattr(hd_profiles, "tc_loader", fixture_load)
        patch.setattr(rf_profiles, "tc_loader", fixture_load)
        scope = dict(namespace, tc_loader=fixture_load, require_saved_input=_saved_input)
        _run(path, prepared, scope)


def _forbid_generation(namespace, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Ordinary comparison cells must only read saved CSVs")

    from HD import tc_generation as hd_generation
    from RF import tc_generation as rf_generation
    from RF import rflocate
    for module, names in (
        (hd_generation, ("generate_hd_tc",)),
        (rf_generation, ("generate_rf_comparison", "generate_rf_projection", "generate_rf_tc")),
        (rflocate, ("load_rfmap", "detect_rf", "save_rf", "save_rf_tc")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, unexpected)
            namespace[name] = unexpected
    namespace["generate_comparison_inputs"] = unexpected


@pytest.mark.parametrize("rf_only", [False, True])
@pytest.mark.parametrize("name", PAIRED_NOTEBOOKS)
def test_paired_notebooks_read_saved_inputs_preserving_ids_and_values(tmp_path, monkeypatch, name, rf_only):
    path, namespace, prepared = _paired_namespace(name, tmp_path, rf_only)
    _fixture_csvs(path, prepared, namespace, monkeypatch)
    saved = {file: file.read_bytes() for file in tmp_path.rglob("*.csv")}
    assert len(saved) >= 10
    _forbid_generation(namespace, monkeypatch)
    _run(path, prepared, namespace)
    normalized = [key for key in namespace if key.startswith("hd_normalized_")]
    assert normalized
    for hd_name in normalized:
        key = hd_name.removeprefix("hd_normalized_")
        hd, rf = namespace[hd_name], namespace[f"rf_normalized_{key}"]
        raw_hd, raw_rf = namespace[f"fm_{key}"], namespace[f"rf_{key}"]
        unit_keys = raw_hd.index.intersection(raw_rf.index).tolist()
        assert hd.index.names == rf.index.names == ["unit_id"]
        assert hd.index.tolist() == rf.index.tolist() == unit_keys
        assert {unit.rsplit(":", 1)[-1] for unit in unit_keys} == {"23", "7"}
        assert hd.shape[1] == rf.shape[1] == 30
        for table, raw, kind in ((hd, raw_hd, "HD"), (rf, raw_rf, "RF")):
            expected = raw.loc[unit_keys].to_numpy().copy()
            if "inhibitory" in name or key == "all":
                if "inhibitory" in name and kind == "RF":
                    expected -= np.nanmin(expected, axis=1)[:, None]
                maximum = np.nanmax(expected, axis=1)
                expected /= np.where(maximum == 0, 1., maximum)[:, None]
            np.testing.assert_allclose(table.to_numpy(), expected, equal_nan=True)
    assert set(tmp_path.rglob("*.*")) == set(saved)
    assert all(file.read_bytes() == contents for file, contents in saved.items())


@pytest.mark.parametrize("rf_only", [False, True])
def test_paired_notebook_keeps_all_mice_distinct_and_pools_each_once(tmp_path, monkeypatch, rf_only):
    path, namespace, prepared = _paired_namespace("tc_comparison_pairs.ipynb", tmp_path, rf_only)
    _fixture_csvs(path, prepared, namespace, monkeypatch)
    _forbid_generation(namespace, monkeypatch)
    _run(path, prepared, namespace)

    recordings = {
        "m14": ("260609", ("A",)),
        "m15": ("260630", ("A", "B")),
        "m19": ("260827", ("A",)),
        "m20": ("260921", ("A",)),
        "m21": ("261006", ("A",)),
        "m22": ("261008", ("A",)),
        "m23": ("261015", ("A",)),
    }
    pooled_keys = []
    for mouse, (date, probes) in recordings.items():
        expected = [f"{mouse}:{date}:{probe}:{unit}" for probe in probes for unit in (23, 7)]
        for prefix in ("hd_normalized_", "rf_normalized_"):
            assert namespace[f"{prefix}{mouse}"].index.tolist() == expected
        pooled_keys.extend(expected)

    for prefix in ("hd_normalized_", "rf_normalized_"):
        pooled = namespace[f"{prefix}all"]
        assert pooled.index.is_unique
        assert pooled.index.tolist() == pooled_keys
        for probe in ("A", "B"):
            assert namespace[f"{prefix}m15_{probe}"].index.tolist() == [
                f"m15:260630:{probe}:{unit}" for unit in (23, 7)
            ]


@pytest.mark.parametrize("missing_kind", ["HD", "RF"])
@pytest.mark.parametrize("name", PAIRED_NOTEBOOKS)
def test_paired_notebooks_missing_input_raises_without_generation(tmp_path, monkeypatch, name, missing_kind):
    path, namespace, prepared = _paired_namespace(name, tmp_path, True)
    _fixture_csvs(path, prepared, namespace, monkeypatch)
    pattern = "hd_*.csv" if missing_kind == "HD" else "rf_*.csv"
    for file in tmp_path.rglob(pattern):
        file.unlink()
    saved = {file: file.read_bytes() for file in tmp_path.rglob("*.csv")}
    _forbid_generation(namespace, monkeypatch)
    with pytest.raises(FileNotFoundError):
        _run(path, prepared, namespace)
    assert {file: file.read_bytes() for file in tmp_path.rglob("*.csv")} == saved


@pytest.mark.parametrize("rf_only", [False, True])
def test_hd_rf_notebook_reads_native_projection_without_raw_sources(tmp_path, rf_only):
    path, trees = _trees("hd_rf_comparison.ipynb")
    blocks = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.For)):
                continue
            start = next((i for i, item in enumerate(node.body)
                          if "rf_csv_stem" in _assigned_names(item)), None)
            if start is None:
                continue
            stop = next(i for i, item in enumerate(node.body)
                        if "native_rf" in _assigned_names(item))
            blocks.append(ast.Module(body=node.body[start:stop + 1], type_ignores=[]))
    assert len(blocks) == 1
    source = tmp_path / "m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeB/regular.rfmap"
    saved_path = source.parents[4] / f"regular_ProbeB_1d{'_rfonly' if rf_only else ''}.csv"
    saved_path.parent.mkdir(parents=True)
    saved_path.write_text("unit_id,-145.0,-135.0\n23,7,9\n7,0,\n")
    namespace = dict(rf_source=source, rf_only=rf_only, probe="B", mouse="m14",
                     rf_range_overrides={"m14": [-150, 150]}, tcRange=comparison.tcRange,
                     load_rf_profiles=rf_profiles.load_rf_profiles)
    before = saved_path.read_bytes()
    _run(path, blocks, namespace)
    table = namespace["native_rf"]
    assert table.index.tolist() == [("B", 23), ("B", 7)]
    np.testing.assert_array_equal(table.columns, [-145., -135.])
    np.testing.assert_allclose(table.to_numpy(), [[7., 9.], [0., np.nan]], equal_nan=True)
    assert saved_path.read_bytes() == before
    assert not source.exists()
    saved_path.unlink()
    with pytest.raises(FileNotFoundError, match="generate_rf_comparison_inputs"):
        _run(path, blocks, namespace)
    assert not saved_path.exists()


def _manual_functions(name, namespace):
    path, trees = _trees(name)
    functions = [node for tree in trees for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith("generate_")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("rf_only", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("name", ("hd_rf_comparison.ipynb", "hd_rf_ebc_comparison.ipynb",
                                  "tc_comparison_pairs_inhibitory.ipynb"))
def test_manual_rf_generators_forward_recording_identity_and_explicit_paths(
    tmp_path, name, rf_only, overwrite,
):
    calls, hd_calls, checked = [], [], []

    def generate_rf(source, output, *, projection_path, detected_rf_path,
                    time_range_s, rf_only, unit_prefix, overwrite=False):
        calls.append(dict(source=source, output=output, projection=projection_path,
                          detection=detected_rf_path, window=time_range_s,
                          rf_only=rf_only, prefix=unit_prefix, overwrite=overwrite))
        return output

    def generate_hd(source, output, *, bins, hd_class, unit_prefix, overwrite=False):
        hd_calls.append((source, output, bins, hd_class, unit_prefix, overwrite))
        return output

    root = tmp_path / "m15/260630"
    session = root / "260630_3"
    namespace = dict(Path=Path, root=root, rf_only=rf_only, rf_response_window_s=(0., .2),
                     rf_result_path=rf_result_path, generate_rf_comparison=generate_rf,
                     generate_hd_tc=generate_hd, require_saved_input=checked.append,
                     hd_shuffles=1000, seed=0)
    if name == "hd_rf_comparison.ipynb":
        namespace.update(mouse="m15", date=260630, base_dir=root.parent,
                         rf_session_id=3, probes=("A", "B"))
    # The EBC notebook derives identity from root; no mouse/date globals exist there.
    _manual_functions(name, namespace)
    sources = [session / "data/rfmapping/good/-100_400_1ms" / f"Probe{probe}"
               / "regular_unitsSpikeCounts_260630_3.rfmap" for probe in ("A", "B")]
    if name == "hd_rf_comparison.ipynb":
        namespace["generate_rf_comparison_inputs"](overwrite=overwrite)
    else:
        for probe, source in zip(("A", "B"), sources, strict=True):
            if name == "hd_rf_ebc_comparison.ipynb":
                result = namespace["generate_rf_comparison_input"](source, probe, overwrite=overwrite)
                assert result == calls[-1]["output"]
            else:
                hd_output, rf_output = namespace["generate_comparison_inputs"](
                    root / "260630_1", session, probe, f"m15:260630:{probe}", overwrite=overwrite,
                )
                assert (hd_output, rf_output) == (hd_calls[-1][1], calls[-1]["output"])
    assert len(calls) == 2
    kind = "inhibitory" if "inhibitory" in name else "excitatory"
    suffix = "_rfonly" if rf_only else ""
    for probe, source, call in zip(("A", "B"), sources, calls, strict=True):
        projection_label = f"{source.stem}_Probe{probe}"
        if kind == "inhibitory" and rf_only:
            projection_label += "_inhibitory"
        assert call == dict(
            source=source,
            output=session / "data/tc_comparison" / f"rf_{kind}_x_2d{suffix}_Probe{probe}.csv",
            projection=session / "data" / f"{projection_label}_1d{suffix}.csv",
            detection=rf_result_path(source, rf_type=kind), window=(0., .2),
            rf_only=rf_only, prefix=f"m15:260630:{probe}", overwrite=overwrite,
        )
    if kind == "inhibitory":
        assert len(hd_calls) == 2
        for probe, call in zip(("A", "B"), hd_calls, strict=True):
            assert call == (
                root / f"260630_1/data/tuning_curves/Probe{probe}/tuning_curves.tc",
                root / f"260630_1/data/tc_comparison/hd_class3_Probe{probe}.csv",
                30, 3, f"m15:260630:{probe}", overwrite,
            )
        assert checked == [output for hd_call, rf_call in zip(hd_calls, calls, strict=True)
                           for output in (hd_call[1], rf_call["output"])]
    assert list(tmp_path.rglob("*")) == []


def test_manual_basler_hd_generator_forwards_geometry_probe_and_shuffle_settings(tmp_path):
    calls = []
    loaded = object()
    result = object()

    def load_basler(session, *, bounds_px, size_cm, probe):
        calls.append(("load", session, bounds_px, size_cm, probe))
        return loaded

    def rebuild(basler, *, shuffles, seed):
        assert basler is loaded
        calls.append(("rebuild", shuffles, seed))
        return result

    session = tmp_path / "m20/260921/260921_11"
    bounds = (370., 920., 210., 760.)
    namespace = dict(session12=session, basler_bounds_px=bounds, basler_size_cm=BASLER_SIZE_CM,
                     hd_shuffles=1000, seed=7, load_basler_session=load_basler,
                     rebuild_basler_hd=rebuild)
    _manual_functions("hd_rf_ebc_comparison.ipynb", namespace)
    generate = namespace["generate_hd12_tuning_curve"]
    assert generate("B") is result
    assert generate("A", shuffles=12, shuffle_seed=19) is result
    assert calls == [("load", session, bounds, 40., "B"), ("rebuild", 1000, 7),
                     ("load", session, bounds, 40., "A"), ("rebuild", 12, 19)]
    assert list(tmp_path.rglob("*")) == []
