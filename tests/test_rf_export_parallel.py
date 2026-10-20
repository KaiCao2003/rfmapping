import xml.etree.ElementTree as ET

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image

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


def test_spawned_exports_match_serial_pixels_and_report_completion(
    tmp_path, analyses_by_probe, capsys,
):
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

    expected = {
        f"rfmap/units/{probe}/7{suffix}"
        for probe in ("A", "B") for suffix in (".png", ".svg")
    }
    for directory in (serial_dir, parallel_dir):
        assert {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*") if path.is_file()
        } == expected
    for probe in ("A", "B"):
        relative = f"rfmap/units/{probe}/7.png"
        with Image.open(serial_dir / relative) as serial, Image.open(parallel_dir / relative) as parallel:
            np.testing.assert_array_equal(np.asarray(parallel), np.asarray(serial))
            rgba = parallel.convert("RGBA")
            assert rgba.getextrema()[3] == (255, 255)
            assert rgba.getpixel((0, 0)) == (255, 255, 255, 255)
        svg = ET.parse(parallel_dir / f"rfmap/units/{probe}/7.svg")
        assert svg.getroot().tag == "{http://www.w3.org/2000/svg}svg"


def test_spawned_export_failure_reaches_caller_without_reporting_completion(
    tmp_path, analyses_by_probe, capsys,
):
    output_dir = tmp_path / "blocked"
    (output_dir / "rfmap/units/A/7.png").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        export_rf_units(
            {"A": analyses_by_probe["A"]}, output_dir, workers=2, show_progress=True,
        )

    assert "100%" not in capsys.readouterr().err
    assert not (output_dir / "rfmap/units/A/7.svg").exists()
