"""The canonical RF package stays separate from its compatibility imports."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("rf_detection", "_detector"),
        ("rf_analysis", "workflow"),
        ("rf_rates", "rates"),
        ("rf_trials", "trials"),
        ("rf_plotting", "plotting"),
    ],
)
def test_compatibility_modules_share_the_canonical_implementation(legacy, canonical):
    assert importlib.import_module(f"Utils.{legacy}") is importlib.import_module(
        f"Utils.rflocate.{canonical}"
    )


def test_core_package_import_needs_neither_plotting_nor_legacy_modules():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from Utils.rflocate import RFMap, RFMapList, RFResult, load_rfmap, load_rf, detect_rf\n"
            "assert 'matplotlib' not in sys.modules\n"
            "assert 'pandas' not in sys.modules\n"
            "assert not {'Utils.rfmap', 'Utils.rf_detection', 'Utils.rf_cache', "
            "'Utils.rf_analysis', 'Utils.rf_plotting', 'Utils.rf_rates', 'Utils.rf_trials'} "
            ".intersection(sys.modules)\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
