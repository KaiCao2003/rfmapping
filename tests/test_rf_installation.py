from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import Utils


def test_rf_core_and_result_cache_without_matplotlib(tmp_path: Path) -> None:
    package_root = Path(Utils.__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        # Use the selected source or installed package, never the working directory.
        sys.path.insert(0, sys.argv[1])
        sys.modules["matplotlib"] = None

        import numpy as np
        import Utils.rf_detection as detection
        from Utils.rf_trials import load_regular_rf_trials
        from Utils.rfmap import asrfmap

        values = np.zeros((3, 5))
        values[1, 1:4] = 10.0
        options = dict(
            is_shuffle=False,
            drop_bins=1,
            cluster_forming_z=1.0,
            wrap_x=False,
            result_path=Path(sys.argv[2]),
            show_progress=False,
        )
        mask = asrfmap(values).rf_2d(**options)
        np.testing.assert_array_equal(mask, values > 0)
        assert options["result_path"].is_file()

        def unexpected_detection(*args, **kwargs):
            raise AssertionError("The saved result should be reused")

        detection.detect_rf = unexpected_detection
        restored = asrfmap(values)
        np.testing.assert_array_equal(restored.rf_2d(**options), mask)
        np.testing.assert_array_equal(restored.rf_1d(axis="x", **options), mask.any(axis=0))
        assert restored.rf_2d(is_center=True, **options).sum() == 1
        """
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_root), str(tmp_path / "result.npz")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
