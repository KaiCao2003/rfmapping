import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source", [
        PROJECT_ROOT / "EBB/spatial_cell_analysis.py",
        PROJECT_ROOT / "scripts/spatial_cell_analysis.py",
        PROJECT_ROOT / "Utils/spatial_cell_analysis.py",
        PROJECT_ROOT / "spatial_cell_analysis.py",
        "/home/kai/scripts/spatial_cell_analysis.py",
])
@pytest.mark.parametrize("entrypoint", ["pipeline", "notebook"])
def test_standard_analysis_and_plot_stage_preserve_units_phase_and_timing(tmp_path, source, entrypoint, monkeypatch):
    if not Path(source).is_file():
        pytest.skip("This production entry is only installed on hhw9l84")
    spec = importlib.util.spec_from_file_location("pipeline_test", source)
    analysis = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, analysis)
    spec.loader.exec_module(analysis)
    analysis = sys.modules[spec.name]
    for name, value in dict(x_min=230, x_max=740, y_min=335, y_max=835,
                            date="261009", recording_number=3, probe_name="B", phase_key="test_phase",
                            cm_per_px=analysis.cm_per_px).items():
        monkeypatch.setattr(analysis, name, value)
    times = np.arange(4) * .02
    pose = pd.DataFrame(dict(frame=np.arange(4), center_x=[383.] * 4, center_y=[515.] * 4,
                            hd_deg=[0., 90., 180., 270.]))
    pose.attrs["exposure_times"] = times.copy()
    source_metadata = dict(selected_interval_s=[0., .06], camera_timing=dict(
        camera_output="basler", ttl_pulse_count=4, measured_rate_hz=50.,
        timestamp_reference="synthetic_saved_frame_times"))
    monkeypatch.setattr(analysis, "load_data", lambda **kwargs: (
        pose, times, times.copy(), np.full(4, 7), [7, 9], source_metadata,
    ))
    from EBB.spatial_cell_results import spatial_result_path
    result_path = spatial_result_path(tmp_path, probe_name="B", phase_key="test_phase")
    result_directory = result_path.parent
    paths = analysis.run_analysis(output=result_path, workers=1)
    assert len(paths) == 4
    counts = load_npz(result_directory / "frame_spike_counts.npz").toarray()
    np.testing.assert_array_equal(counts, [[1, 1, 1, 1], [0, 0, 0, 0]])
    with np.load(result_directory / "restored_maps.npz") as saved:
        np.testing.assert_array_equal(saved["frame_times"], times)
        np.testing.assert_array_equal(saved["frame_ids"], np.arange(4))
        np.testing.assert_array_equal(saved["unit_ids"], [7, 9])
        assert saved["allocentric_spike_maps"].sum() == 4
        assert saved["egocentric_occupancy"].sum() > 0
    plot_directory = tmp_path / "data/spatial_cells/ProbeB/all_units"
    hd_tc_path = tmp_path / "data/tuning_curves/ProbeB/tuning_curves.tc"
    hd_tc_path.parent.mkdir(parents=True)
    hd_tc_path.write_text(json.dumps({
        "unit_id": [7], "spike_counts": [np.arange(180).tolist()],
        "occupancy_time_s": np.ones(180).tolist(),
        "angle_bin_edges_deg": np.linspace(0, 360, 181).tolist(),
        "unit_data": {"hd_class": [3]},
    }))
    input_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*result_directory.iterdir(), hd_tc_path]}
    if entrypoint == "pipeline":
        subprocess.run([sys.executable, "-m", "scripts.plot_spatial_cells", str(result_directory), "--workers", "2"], check=True)
    else:
        nbformat = pytest.importorskip("nbformat")
        NotebookClient = pytest.importorskip("nbclient").NotebookClient
        nb = nbformat.read(PROJECT_ROOT / "spatial_cell_plotting.ipynb", as_version=4)
        nb.cells = [c for c in nb.cells if c.id in {"imports", "parameters", "export-all-units"}]
        parameters = next(c for c in nb.cells if c.id == "parameters")
        parameters.source = (
            f"result_path = Path({str(result_path)!r})\n"
            f"plot_directory = Path({str(plot_directory)!r})\nplot_workers = 1\n"
            f"hd_tc_path = Path({str(hd_tc_path)!r})\n"
        )
        client = NotebookClient(
            nb, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv[0] = sys.executable
        client.execute()
    manifest = json.loads((plot_directory / "spatial_plot_manifest.json").read_text())
    assert manifest["unit_count"] == 2 and manifest["panels_per_unit"] == 10
    assert manifest["metadata"]["probe_name"] == "B"
    assert manifest["metadata"]["phase_key"] == "test_phase"
    assert {Path(p["svg_path"]).name for p in manifest["plots"]} == {"unit_7.svg", "unit_9.svg"}
    for extension in ("png", "svg"):
        assert {p.name for p in plot_directory.glob(f"*.{extension}")} == {f"unit_7.{extension}", f"unit_9.{extension}"}
    for svg in plot_directory.glob("*.svg"):
        root = ElementTree.parse(svg).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        meshes = [element for element in root.iter() if element.get("id", "").startswith("QuadMesh_")]
        assert meshes and all(element.get("shape-rendering") == "crispEdges" for element in meshes)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_hashes} == input_hashes
