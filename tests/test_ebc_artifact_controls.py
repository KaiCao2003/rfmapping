"""Check the notebook's spatial bins without running recording controls."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse import csr_matrix

from EBB.ebc_analysis import ray_histogram_matrix, rectangle_distances
from EBB.ebc_geometry import BASLER_SIZE_CM


def test_controls_match_histogram_bins_at_inclusive_40_cm_edges():
    path = Path(__file__).resolve().parents[1] / "ebc_artifact_controls.ipynb"
    cells = json.loads(path.read_text())["cells"]
    namespace = dict(np=np, BASLER_SIZE_CM=BASLER_SIZE_CM, csr_matrix=csr_matrix,
                     gaussian_filter=gaussian_filter, gaussian_filter1d=gaussian_filter1d,
                     rectangle_distances=rectangle_distances, ray_histogram_matrix=ray_histogram_matrix)
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("BANDS12 =") or source.startswith("def prepare_controls("):
            exec(compile(source, str(path), "exec"), namespace)
    xy = np.array([[0., 0.], [40., 40.], [0., 40.], [40., 0.],
                   [2., 2.], [39.999, 39.999], [41., 20.]])
    raw = dict(xy=xy, times=np.arange(len(xy)) * .02, hd=np.zeros(len(xy)),
               counts=np.ones((1, len(xy)), dtype=int), unit_ids=np.array([7]), dt=.02)
    data = namespace["prepare_controls"](raw)
    projection = data["position_projection"]
    assert projection.shape == (400, 6)
    assert data["excluded_geometry_frames"] == 1
    expected, _, _ = np.histogram2d(xy[:6, 1], xy[:6, 0], bins=np.linspace(0., 40., 21))
    occupancy_counts = np.asarray(projection.sum(axis=1)).ravel()
    np.testing.assert_array_equal(occupancy_counts.reshape(20, 20), expected)
    np.testing.assert_array_equal(projection[:, :4].nonzero()[0], [0, 19, 380, 399])
    states = namespace["behavior_states"](data)
    assert states["keys"][:, :2].max() == 19
    information = namespace["spatial_information"](occupancy_counts[None], occupancy_counts * .02)
    np.testing.assert_allclose(information, [0.], atol=1e-12)
