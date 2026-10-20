import numpy as np
import pandas as pd
import pytest

from EBB.ebc_pose import hold_last_in_bounds, hold_rectangle_positions, load_basler_position


@pytest.mark.parametrize("values, expected, held_indices", [
    ([6, 7, 8, 9, 15, 8, 7, 6], [6, 7, 8, 9, 9, 8, 7, 6], [4]),
    ([1, 2, 3, 4, 19, 18, 17, 7, 8], [1, 2, 3, 4, 4, 4, 4, 7, 8], [4, 5, 6]),
])
def test_user_example_holds_until_reentry(values, expected, held_indices):
    raw = np.asarray(values, dtype=float)[:, None]
    corrected, held, initial = hold_last_in_bounds(raw, (raw[:, 0] >= 0) & (raw[:, 0] <= 10))
    np.testing.assert_array_equal(corrected[:, 0], expected)
    np.testing.assert_array_equal(np.flatnonzero(held), held_indices)
    assert not initial.any()


def test_whole_point_leading_outside_missing_and_inclusive_edges():
    raw = np.array([[11, 2], [1, 2], [3, 11], [np.nan, 5], [-1, 4], [10, 0], [6, 7.]])
    corrected, held, initial = hold_rectangle_positions(raw, [0, 10, 0, 10])
    np.testing.assert_equal(corrected, [[np.nan, np.nan], [1, 2], [1, 2],
                                      [np.nan, 5], [1, 2], [10, 0], [6, 7]])
    np.testing.assert_array_equal(np.flatnonzero(held), [2, 4])
    np.testing.assert_array_equal(np.flatnonzero(initial), [0])
    np.testing.assert_equal(raw[2], [3, 11])


def test_basler_keeps_heading_clock_missing_rows_and_provenance(tmp_path):
    path = tmp_path / "pose.csv"
    pd.DataFrame(dict(frame=[0, 1, 3, 4], center_x=[11., 2., 12., 5.],
                      center_y=[2., 3., 4., 6.], hd_deg=[0., 90., 180., 270.])).to_csv(path, index=False)
    loaded = load_basler_position(path, np.arange(5) / 50, bounds_px=[0, 10, 0, 10],
                                  size_cm=10, heading_clockwise=True)
    np.testing.assert_equal(loaded.xyz_cm[:, :2], [[np.nan, np.nan], [2, 7],
                                                [np.nan, np.nan], [2, 7], [5, 4]])
    np.testing.assert_equal(loaded.hd_deg, [90, 0, np.nan, 270, 180])
    np.testing.assert_equal(loaded.times_s, np.arange(5) / 50)
    np.testing.assert_array_equal(np.flatnonzero(loaded.position_held), [3])
    np.testing.assert_array_equal(np.flatnonzero(loaded.initial_outside), [0])
    np.testing.assert_equal(loaded.raw_xyz_cm[3], [12, 6, 0])


def test_motive_hold_precedes_video_mapping_and_holds_xyz(tmp_path):
    from EBB.ebc_pose import load_motive_position
    path = tmp_path / "pose.csv"
    columns = pd.MultiIndex.from_tuples([("meta", "id", "meta", "Frame")]
        + [("chosen", "rigid", kind, axis) for kind in ("Position", "Rotation") for axis in "XYZ"])
    pd.DataFrame([[0, 1, 0, 5, 0, 0, 0], [1, 2, 0, 6, 0, 0, 90],
                  [2, 20, 0, 9, 0, 0, 180]], columns=columns).to_csv(path, index=False)
    loaded = load_motive_position(path, [0., .04], [0, 2], units_per_cm=1,
                                  reference_quaternion_xyzw=[0, 0, 0, 1], hd_world_zero_deg=0,
                                  headplate="chosen", center_xy_cm=[0, 0], radius_cm=10)
    np.testing.assert_equal(loaded.xyz_cm, [[1, 0, 5], [2, 0, 6]])
    np.testing.assert_equal(loaded.raw_xyz_cm, [[1, 0, 5], [20, 0, 9]])
    np.testing.assert_array_equal(loaded.position_held, [False, True])
    np.testing.assert_allclose(loaded.hd_deg, [0, 180])


def test_spatial_loader_holds_before_phase_without_changing_frame_selection(tmp_path, monkeypatch):
    from EBB import spatial_cell_analysis as spatial
    session = tmp_path / "261009_3"
    data = session / "data"
    sort = session / "kilosort/ProbeA/kilosort_test"
    sort.mkdir(parents=True)
    (data / "probeA").mkdir(parents=True)
    pd.DataFrame(dict(frame=np.arange(5), center_x=[11., 2., 12., np.nan, 7.],
                      center_y=[3., 4., 5., 6., 8.], hd_deg=np.arange(5)*30.)).to_csv(session / "261009.csv", index=False)
    pd.DataFrame(dict(interval_type=["phase"], start=[.04], end=[.08])).to_csv(data / "interval_table.csv", index=False)
    np.save(sort / "spike_clusters.npy", [7, 7])
    np.save(data / "probeA/adc_spike_time.npy", [.04, .08])
    monkeypatch.setattr(spatial, "read_formatted_json", lambda path: {"session_info": {}})
    monkeypatch.setattr(spatial, "get_exposure_timestamps", lambda **kwargs: (np.arange(5)*.02, 0., {}))
    monkeypatch.setattr(spatial, "read_good_unit_ids", lambda path: [7])
    pose,times,spikes,clusters,units,source = spatial.load_data(
        session_dir=session, probe="A", phase="phase", bounds_px=[0,10,0,10])
    np.testing.assert_array_equal(pose.frame, [2,4])
    np.testing.assert_array_equal(times, [.04,.08])
    np.testing.assert_array_equal(spikes, [.04,.08])
    np.testing.assert_array_equal(pose[["center_x","center_y"]], [[2,4],[7,8]])
    np.testing.assert_array_equal(pose[["raw_center_x","raw_center_y"]], [[12,5],[7,8]])
    np.testing.assert_array_equal(pose.hd_deg, [60,120])
    np.testing.assert_array_equal(pose.position_held, [True,False])
