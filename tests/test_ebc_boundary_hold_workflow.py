import numpy as np
import pandas as pd

from EBB.ebc_geometry import rectangle_boundary
from EBB.ebc_pose import load_basler_position
from EBB.ebc_workflow import prepare_video_data, save_geometry_audit


def test_held_position_is_used_by_geometry_and_saved_with_raw_coordinates(tmp_path):
    pose = tmp_path / 'pose.csv'
    pd.DataFrame(dict(frame=np.arange(5), center_x=[3, 12, 13, 8, 7],
                      center_y=[4, 5, 6, 6, 5], hd_deg=[0, 90, 180, 270, 0])).to_csv(pose, index=False)
    times = np.arange(5) / 50
    position = load_basler_position(pose, times, bounds_px=(0, 10, 0, 10), size_cm=10,
                                   heading_clockwise=True)
    boundary = rectangle_boundary((0, 10, 0, 10), 10)
    config = dict(session=tmp_path, probe='A', kilosort_dir=tmp_path, interval_s=[0, 1])
    data = prepare_video_data(config, position, boundary, times, 0, {})
    data['source_frame_ids'] = position.frame_ids
    np.testing.assert_array_equal(data['xy'], [[3, 6], [3, 6], [3, 6], [8, 4], [7, 5]])
    np.testing.assert_array_equal(data['position_held'], [False, True, True, False, False])
    np.testing.assert_array_equal(data['hd'], [90, 0, 270, 180, 90])
    assert data['ebc_info'].valid.all()
    # Holding position does not freeze the independently measured head direction.
    np.testing.assert_allclose(data['ebc_info'].distances_cm[:3, 0], [4, 7, 6])
    result = dict(source_first_frame=0, source_last_frame=4)
    destination = tmp_path / 'geometry.csv'
    save_geometry_audit(data, result, destination)
    saved = pd.read_csv(destination)
    np.testing.assert_array_equal(saved.position_held, data['position_held'])
    np.testing.assert_array_equal(saved.raw_x_cm, [3, 12, 13, 8, 7])
    np.testing.assert_array_equal(saved.x_cm, [3, 3, 3, 8, 7])
    np.testing.assert_allclose(saved.adc_time_s, times)
