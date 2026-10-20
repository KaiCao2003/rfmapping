"""The four entrypoints compile; plotting needs only a saved analysis run."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from EBB.center_cell_io import save_center_analysis
from EBB.center_cells import CenterCellConfig, analyze_center_cells, prepare_center_behavior
from Utils.theta_analysis import analyze_theta_units, extract_theta_phase, summarize_lfp_reference
from Utils.theta_io import load_theta_analysis, save_theta_analysis


ROOT = Path(__file__).resolve().parents[1]


def notebook_trees(name):
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
            yield ast.parse(source, name)


@pytest.mark.parametrize("name", [
    "cb_cd_analysis.ipynb", "cb_cd_plotting.ipynb", "theta_analysis.ipynb", "theta_plotting.ipynb",
])
def test_notebook_stage_boundaries(name):
    trees = list(notebook_trees(name))
    for tree in trees:
        compile(tree, name, "exec")
    modules = [node.module for tree in trees for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    if name.endswith("plotting.ipynb"):
        imported = {alias.name for tree in trees for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert not any(name.startswith(("analyze_", "generate_", "extract_", "prepare_"))
                       for name in imported)
        assert not imported.intersection({"load_center_session", "load_motive_center_session", "load_lfp"})
    else:
        assert not any(module and module.endswith("plotting") for module in modules)
    if name.startswith("cb_cd"):
        assert not any(module and ".theta" in module for module in modules)
    else:
        assert not any(alias.name == "analyze_center_cells" for tree in trees for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) for alias in node.names)


@pytest.fixture
def saved_runs(tmp_path):
    times = np.arange(0., 60., .05)
    angle = times * .5
    radius = 8 + 6 * np.sin(times)
    xy = 20 + radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    behavior = prepare_center_behavior(
        times, xy, times * 51 % 360, 40.,
        config=CenterCellConfig(position_smoothing_s=0., min_spikes_exclusive=5,
                                min_spatial_coverage_exclusive=0.),
    )
    center = analyze_center_cells(behavior, {0: times[::3], 9: np.empty(0)}, n_shuffles=2)
    center_dir = save_center_analysis(tmp_path / "center", center, provenance={})

    lfp_times = np.arange(0., 8., .002)
    lfp = dict(times_s=lfp_times, voltage_uv=np.cos(2 * np.pi * 8 * lfp_times), sample_rate_hz=500.)
    phase = extract_theta_phase(lfp_times, lfp["voltage_uv"], sample_rate_hz=500.)
    intervals = np.array([[0., 8.]])
    theta = analyze_theta_units(
        {0: np.arange(1.03125, 6.5, .125), 9: np.empty(0)}, phase,
        tracking_intervals_s=intervals, moving_intervals_s=intervals, min_spikes_exclusive=10,
    )
    theta["metadata"]["reference_channel"] = 0
    theta_dir = save_theta_analysis(
        tmp_path / "theta", theta, provenance={},
        lfp_reference=summarize_lfp_reference(lfp, phase, intervals),
    )
    return dict(cb_cd=center_dir, theta=theta_dir)


@pytest.mark.parametrize("workflow,with_reference", [("cb_cd", True), ("theta", True), ("theta", False)])
def test_plotting_notebook_exports_from_saved_results_only(saved_runs, workflow, with_reference, monkeypatch):
    from EBB import center_cell_io, center_cells
    from Utils import theta_analysis

    def forbidden(*args, **kwargs):
        pytest.fail("Saved plotting must not load raw data or rerun analysis")

    for module, names in (
        (center_cell_io, ("load_center_session", "load_motive_center_session", "load_lfp")),
        (center_cells, ("analyze_center_cells", "analyze_center_theta", "prepare_center_behavior")),
        (theta_analysis, ("analyze_theta_units", "extract_theta_phase", "theta_rhythmicity")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)

    directory = saved_runs[workflow]
    if not with_reference:
        saved = load_theta_analysis(directory)
        saved["metadata"].pop("reference_channel")
        save_theta_analysis(directory, saved, provenance={}, overwrite=True)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    overrides = dict(result_dir=directory, plot_directory=directory / "figures", unit_id=0, is_save=True)
    namespace = {"_overrides": overrides}
    name = f"{workflow}_plotting.ipynb"
    try:
        for tree in notebook_trees(name):
            for node in tree.body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in overrides):
                    node.value = ast.parse(f"_overrides[{node.targets[0].id!r}]", mode="eval").body
            exec(compile(ast.fix_missing_locations(tree), name, "exec"), namespace)
        assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == original
        for unit in (0, 9):
            for extension in ("png", "svg"):
                assert (directory / "figures" / "all_units" / f"unit_{unit}.{extension}").stat().st_size > 0
        assert not plt.get_fignums()
    finally:
        plt.close("all")
