import numpy as np
import pytest
from matplotlib import pyplot as plt

from RF.rflocate.plotting import export_rf_units
from RF.rflocate.workflow import analyze_rf_file
from test_rf_analysis import _write_indexed_source


@pytest.fixture
def analyses_by_probe(tmp_path):
    analyses = {}
    for probe, start in (("A", 5), ("B", 17)):
        counts = np.full((1, 7, 30, 3), 4, dtype=np.uint32)
        counts[0, 2:5, start:start + 4, 1:] = 24
        source = tmp_path / f"{probe}.rfmap"
        _write_indexed_source(source, counts=counts, unit_ids=[7])
        analyses[probe] = {
            "excitatory": analyze_rf_file(source, probe=probe, save_results=False, drop_bins=2, reference_2d="global"),
        }
    return analyses


def test_spawned_exports_match_serial_pdfs_and_report_completion(
    tmp_path, analyses_by_probe, capsys, monkeypatch,
):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    open_figures = set(plt.get_fignums())
    assert export_rf_units(
        analyses_by_probe, serial_dir, workers=1, show_progress=False, rf_only=True,
    ) == 2
    assert "RF plots" not in capsys.readouterr().err

    assert export_rf_units(
        analyses_by_probe, parallel_dir, workers=2, show_progress=True, rf_only=True,
    ) == 2
    progress = capsys.readouterr().err
    assert "RF plots (excitatory)" in progress
    assert "2/2" in progress
    assert "100%" in progress
    assert set(plt.get_fignums()) == open_figures

    expected = {f"Probe{probe}/rfmap/ext/unit_7.pdf" for probe in ("A", "B")}
    for directory in (serial_dir, parallel_dir):
        assert {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        } == expected
    for relative in expected:
        serial = (serial_dir / relative).read_bytes()
        assert serial.startswith(b"%PDF-")
        assert (parallel_dir / relative).read_bytes() == serial


def test_spawned_export_failure_reaches_caller_without_reporting_completion(
    tmp_path, analyses_by_probe, capsys,
):
    output_dir = tmp_path / "blocked"
    (output_dir / "rfmap/ext/unit_7.pdf").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        export_rf_units(
            {"A": analyses_by_probe["A"]}, output_dir, workers=2, show_progress=True,
        )

    assert "100%" not in capsys.readouterr().err
    assert not any(path.is_file() for path in output_dir.rglob("*.pdf"))
