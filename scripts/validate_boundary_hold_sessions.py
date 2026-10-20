"""Independently verify held locations and raw spatial bin assignments for six sessions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz


COHORTS = ("m22_261009_3", "m22_261008_11", "m21_261006_11",
           "m20_260921_11", "m20_260918_11", "m19_260831_2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_hold(points, bounds):
    """Serial reference intentionally independent of the shared vectorized helper."""
    left, right, top, bottom = bounds
    corrected = points.copy()
    held = np.zeros(len(points), dtype=bool)
    initial = held.copy()
    donor = np.full(len(points), -1, dtype=int)
    last = -1
    for index, point in enumerate(points):
        if not np.isfinite(point).all():
            continue
        x, y = point
        if left <= x <= right and top <= y <= bottom:
            last = index
        elif last < 0:
            corrected[index] = np.nan
            initial[index] = True
        else:
            corrected[index] = points[last]
            held[index] = True
            donor[index] = last
    return corrected, held, initial, donor


def projection(first, second, first_edges, second_edges):
    """Independent histogram2d binning, including the final right edge."""
    a, b = np.broadcast_arrays(first, second)
    ia = np.searchsorted(first_edges, a, side="right") - 1
    ib = np.searchsorted(second_edges, b, side="right") - 1
    ia[a == first_edges[-1]] -= 1
    ib[b == second_edges[-1]] -= 1
    na, nb = len(first_edges) - 1, len(second_edges) - 1
    valid = np.isfinite(a) & np.isfinite(b) & (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    frames = np.broadcast_to(np.arange(len(a)).reshape((-1,) + (1,) * (a.ndim - 1)), a.shape)
    return csr_matrix((np.ones(int(valid.sum()), dtype=np.int64),
                       (ia[valid] * nb + ib[valid], frames[valid])), shape=(na * nb, len(a)))


def ray_distances(xy_px, hd, bearings, bounds):
    left, right, top, bottom = bounds
    x = (xy_px[:, 0] - left) * 41. / (right - left)
    y = (bottom - xy_px[:, 1]) * 41. / (bottom - top)
    angle = np.deg2rad(90. - hd[:, None] - bearings)
    dx, dy = np.cos(angle), np.sin(angle)
    tx = np.divide(np.where(dx > 0, 41. - x[:, None], -x[:, None]), dx,
                   out=np.full_like(dx, np.inf), where=np.abs(dx) > 1e-12)
    ty = np.divide(np.where(dy > 0, 41. - y[:, None], -y[:, None]), dy,
                   out=np.full_like(dy, np.inf), where=np.abs(dy) > 1e-12)
    distances = np.maximum(np.minimum(tx, ty), 0.)
    valid = np.isfinite(xy_px).all(axis=1) & np.isfinite(hd)
    valid &= (x >= 0) & (x <= 41) & (y >= 0) & (y <= 41)
    distances[~valid] = np.nan
    return distances


def verify(cohort, output_root, previous_root):
    directory, previous = output_root / cohort, previous_root / cohort
    audit = json.loads((directory / "audit.json").read_text())
    inputs = [directory / name for name in ("audit.json", "restored_maps.npz", "position_correction.npz", "frame_spike_counts.npz")]
    inputs.append(Path(audit["source"]["pose_path"]))
    before = {str(path): sha(path) for path in inputs}
    assert before[str(inputs[-1])] == audit["source_pose_sha256"]
    bounds = audit["bounds_px"]
    pose = pd.read_csv(inputs[-1], usecols=["frame", "center_x", "center_y", "hd_deg"])
    assert pose.frame.is_unique and np.all(np.diff(pose.frame) > 0)
    raw = pose[["center_x", "center_y"]].to_numpy(float)
    corrected, held, initial, donor = reference_hold(raw, bounds)
    saved = dict(np.load(directory / "restored_maps.npz"))
    positions = dict(np.load(directory / "position_correction.npz"))
    old = dict(np.load(previous / "restored_maps.npz"))
    ids = positions["frame_ids"]
    rows = pd.Index(pose.frame).get_indexer(ids)
    assert (rows >= 0).all()
    np.testing.assert_array_equal(positions["raw_xy_px"], raw[rows])
    np.testing.assert_array_equal(positions["corrected_xy_px"], corrected[rows])
    np.testing.assert_array_equal(positions["position_held"], held[rows])
    np.testing.assert_array_equal(positions["initial_outside"], initial[rows])
    np.testing.assert_array_equal(positions["hd_deg"], pose.hd_deg.to_numpy()[rows])
    np.testing.assert_array_equal(saved["frame_ids"], ids)
    np.testing.assert_array_equal(saved["frame_times"], positions["frame_times"])
    for key in ("frame_times", "unit_ids", "theta_edges", "distance_edges", "x_edges", "y_edges", "allocentric_tuning_curves"):
        np.testing.assert_array_equal(saved[key], old[key])
    counts = load_npz(directory / "frame_spike_counts.npz")
    old_counts = load_npz(previous / "frame_spike_counts.npz")
    assert counts.shape == old_counts.shape and (counts != old_counts).nnz == 0
    left, right, top, bottom = bounds
    xy = corrected[rows]
    x = (xy[:, 0] - left) * 41. / (right - left)
    y = (xy[:, 1] - top) * 41. / (bottom - top)
    np.testing.assert_array_equal(saved["trajectory_x_cm"], x)
    np.testing.assert_array_equal(saved["trajectory_y_cm"], y)
    dt = float(np.median(np.diff(saved["frame_times"])))
    xy_projection = projection(x, y, saved["x_edges"], saved["y_edges"])
    xy_occupancy = np.asarray(xy_projection.sum(axis=1)).ravel().reshape(40, 40) * dt
    xy_spikes = (xy_projection @ counts.T).toarray().T.reshape(len(saved["unit_ids"]), 40, 40)
    np.testing.assert_allclose(saved["allocentric_occupancy"], xy_occupancy, rtol=0., atol=5e-8)
    np.testing.assert_array_equal(saved["allocentric_spike_maps"], xy_spikes)
    angles = (saved["theta_edges"][:-1] + saved["theta_edges"][1:]) / 2
    shape = len(angles), len(saved["distance_edges"]) - 1
    ego_occupancy = np.zeros(shape)
    ego_spikes = np.zeros((len(saved["unit_ids"]), *shape), dtype=np.int64)
    for first in range(0, len(ids), 5000):
        stop = min(first + 5000, len(ids))
        distances = ray_distances(xy[first:stop], positions["hd_deg"][first:stop], angles, bounds)
        rays = projection(np.broadcast_to(angles, distances.shape), distances,
                          saved["theta_edges"], saved["distance_edges"])
        ego_occupancy += np.asarray(rays.sum(axis=1)).ravel().reshape(shape) * dt
        ego_spikes += (rays @ counts[:, first:stop].T).toarray().T.reshape(ego_spikes.shape)
    np.testing.assert_allclose(saved["egocentric_occupancy"], ego_occupancy, rtol=0., atol=5e-8)
    np.testing.assert_array_equal(saved["egocentric_spike_maps"], ego_spikes)
    held_spikes = int(counts[:, held[rows]].sum())
    allocentric_added = int(saved["allocentric_spike_maps"].sum() - old["allocentric_spike_maps"].sum())
    assert allocentric_added == held_spikes
    np.testing.assert_allclose(saved["allocentric_occupancy"].sum() - old["allocentric_occupancy"].sum(),
                               int(held[rows].sum()) * dt, rtol=0., atol=5e-8)
    assert before == {str(path): sha(path) for path in inputs}
    report = dict(cohort=cohort, status="passed", frames=len(ids), units=len(saved["unit_ids"]), bounds_px=bounds,
                  held_frames=int(held[rows].sum()), initial_outside_frames=int(initial[rows].sum()),
                  source_missing_rows=int((~np.isfinite(raw).all(axis=1)).sum()),
                  donor_before_selected_phase=int(np.sum(held[rows] & (donor[rows] < rows[0]))),
                  spikes_on_held_frames=held_spikes, allocentric_added_spikes=allocentric_added,
                  allocentric_added_occupancy_s=float(int(held[rows].sum()) * dt),
                  ebc_added_ray_spike_contributions=int(ego_spikes.sum() - old["egocentric_spike_maps"].sum()),
                  ebc_added_ray_occupancy_s=float(ego_occupancy.sum() - old["egocentric_occupancy"].sum()),
                  all_occupancy_bins_and_all_units_spike_maps_match=True,
                  hd_times_units_native_grids_and_frame_spike_assignments_unchanged=True,
                  raw_pose_unchanged=True, input_sha256=before)
    (directory / "independent_boundary_hold_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "input_sha256"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output/session_comparisons_20261012/spatial_ebc/boundary_hold"))
    parser.add_argument("--previous-root", type=Path, default=Path("output/session_comparisons_20261012/spatial_ebc/clockwise"))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    args = parser.parse_args()
    for cohort in args.cohorts:
        verify(cohort, args.output_root, args.previous_root)
    reports = []
    for cohort in COHORTS:
        path = args.output_root / cohort / "independent_boundary_hold_validation.json"
        if path.exists():
            report = json.loads(path.read_text())
            assert all(sha(name) == digest for name, digest in report["input_sha256"].items())
            reports.append(report)
    summary = dict(status="passed", all_six_passed=len(reports) == 6, cohorts=reports,
                   cohorts_validated_this_run=list(args.cohorts), validation_script_sha256=sha(__file__),
                   reference="Independent serial hold and physical-CW ray histogram; no shared production geometry helpers")
    (args.output_root / "independent_boundary_hold_validation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
