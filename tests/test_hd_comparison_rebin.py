import json

import numpy as np
import pytest

from HD.profiles import hd_profiles, load_hd_comparison, prep_hd_tc
from HD.tc_generation import generate_hd_tc
from Utils.direction_comparison import concat


def test_legacy_hd_csv_uses_matching_saved_exposure_for_coarsening(tmp_path):
    folder = tmp_path / "m1/260101/260101_1/data"
    source = folder / "tuning_curves/ProbeA/tuning_curves.tc"
    source.parent.mkdir(parents=True)
    data = dict(unit_id=[7], unit_data={"hd_class": [3]},
                occupancy_time_s=[1, 9, 1, 1, 0, 1, 0, 0],
                spike_counts=[[10, 0, 3, 3, 0, 0, 0, 0]])
    source.write_text(json.dumps(data))
    generate_hd_tc(source, folder / "tc_comparison/hd_class3_ProbeA.csv",
                   bins=8, hd_class=3, unit_prefix="old:prefix")

    def load():
        return load_hd_comparison("m1", tmp_path, "A", "260101", 1)

    table = load()
    coarse = prep_hd_tc(table, bins=4)
    np.testing.assert_allclose(coarse, [[1 / 3, 1., 0., np.nan]], equal_nan=True)
    np.testing.assert_array_equal(coarse.columns, [45., 135., -135., -45.])
    assert coarse.index.tolist() == ["old:prefix:7"]
    assert coarse.attrs["range"] == table.attrs["range"]

    # A regenerated TC with different rates no longer describes this CSV.
    data["occupancy_time_s"][0] = 2
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exposure"):
        prep_hd_tc(load(), bins=4)
    source.unlink()
    same_grid = prep_hd_tc(load(), bins=8)
    np.testing.assert_allclose(same_grid, [[1., 0., .3, .3, np.nan, 0., np.nan, np.nan]])
    with pytest.raises(ValueError, match="exposure"):
        prep_hd_tc(load(), bins=4)


def test_pooled_hd_tables_do_not_borrow_the_first_recordings_exposure():
    data = dict(unit_id=[7], unit_data={"hd_class": [3]},
                occupancy_time_s=[1, 4, 2, 1], spike_counts=[[1, 4, 2, 1]])
    first = hd_profiles(data, bins=4)
    second = hd_profiles(dict(data, unit_id=[9], occupancy_time_s=[4, 1, 2, 1]), bins=4)
    with pytest.raises(ValueError, match="exposure"):
        prep_hd_tc(concat(first, second), bins=2)
    prepared = concat(prep_hd_tc(first, bins=2), prep_hd_tc(second, bins=2))
    np.testing.assert_allclose(prepared, np.ones((2, 2)))
