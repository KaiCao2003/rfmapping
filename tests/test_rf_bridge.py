"""Regression coverage for the automatic RF detection caller."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_rf_analysis import _write_indexed_source


def test_cli_signed_thresholds_refresh_both_comparison_tables(tmp_path):
    source = tmp_path / "bridge input 'quoted'.rfmap"
    _write_indexed_source(source)
    comparison_dir = tmp_path / "data" / "tc_comparison"
    comparison_dir.mkdir(parents=True)
    for kind in ("excitatory", "inhibitory"):
        (comparison_dir / f"rf_{kind}_x_2d_rfonly_ProbeB.csv").write_text("stale comparison\n")
        suffix = "_inhibitory" if kind == "inhibitory" else ""
        (comparison_dir.parent / f"{source.stem}_ProbeB{suffix}_1d_rfonly.csv").write_text("stale projection\n")
    script = Path(__file__).resolve().parents[1] / "locate_rf.py"
    subprocess.run(
        [sys.executable, str(script), str(source), "--probe", "B",
         "--sigma-2d", "1.2", "-0.8", "--sigma-1d", "0.9", "-0.5",
         "--comparison-output-dir", str(comparison_dir), "--unit-prefix", "m22:261008:B",
         "--overwrite-comparison", "--no-plots", "--drop-bins", "2",
         "--reference-2d", "global"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    for kind, expected_sigmas in (("excitatory", (1.2, 0.9)), ("inhibitory", (-0.8, -0.5))):
        suffix = "_inhibitory" if kind == "inhibitory" else ""
        summary = json.loads(source.with_name(f"{source.stem}{suffix}_analysis.json").read_text())
        assert (summary["parameters"]["sigma_2d"], summary["parameters"]["sigma_1d"]) == expected_sigmas
        assert "stale" not in (comparison_dir / f"rf_{kind}_x_2d_rfonly_ProbeB.csv").read_text()
        assert "stale" not in (comparison_dir.parent / f"{source.stem}_ProbeB{suffix}_1d_rfonly.csv").read_text()
        for dimension in ("", "_1d"):
            with np.load(source.with_name(f"{source.stem}{suffix}{dimension}.npz")) as result:
                assert result["mask_2d"].ndim == 3


@pytest.mark.parametrize("module_name, arguments, operation", [
    ("locate_rf", ["missing.rfmap", "--probe", "A"], "analyze_rf_file"),
    ("scripts.redetect_rf_sessions", ["missing_session"], "redetect_session"),
])
@pytest.mark.parametrize("missing_flag, supplied_options", [
    ("--drop-bins", ["--reference-2d", "global"]),
    ("--reference-2d", ["--drop-bins", "0"]),
])
def test_cli_requires_both_detection_choices_before_io(
    tmp_path, monkeypatch, capsys, module_name, arguments, operation,
    missing_flag, supplied_options,
):
    module = importlib.import_module(module_name)

    def unexpected_analysis(*args, **kwargs):
        pytest.fail(f"Missing {missing_flag} must fail before reading or analyzing sources")

    monkeypatch.setattr(module, operation, unexpected_analysis)
    monkeypatch.setattr(sys, "argv", [module_name, *arguments, *supplied_options])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert f"required: {missing_flag}" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
