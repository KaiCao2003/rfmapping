import json
from pathlib import Path

import numpy as np
import pandas as pd

import spatial_cell_analysis as spatial
from Utils.ebc_analysis import (
    circle_distances, rectangle_distances, ray_histogram_matrix, band_mask,
    distance_correlations, row_spearman, ebc_distance_peaks, rates_from_counts, bootstrap_bounds,
    frame_counts,
)

# Exercise the notebook's actual definitions without running its data loads,
# figures, simulations, or exports at test collection time.
_notebook_path = Path(__file__).resolve().parents[1] / "ebc_artifact_controls.ipynb"
_notebook = json.loads(_notebook_path.read_text())
_controls = {}
for _cell in _notebook["cells"]:
    if _cell["id"] in {"cell-01", "ebc-analysis-data", "ebc-analysis-statistics"}:
        _source = "".join(line for line in _cell["source"] if not line.startswith("%"))
        exec(compile(_source, f"{_notebook_path}:{_cell['id']}", "exec"), _controls)
for _name in ("prepare_controls", "behavior_states", "match_states", "matched_firing_rates",
              "hd_expected_counts", "hd_only_controls", "spatial_information"):
    globals()[_name] = _controls[_name]


def test_circle_intersections_at_center_and_off_center():
    xy = np.array([[0., 0.], [5., 0.]])
    angles = np.array([0., 90., 180., 270.])
    inner = circle_distances(xy, np.zeros(2), 15, angles)
    outer = circle_distances(xy, np.zeros(2), 30, angles)
    np.testing.assert_allclose(inner[0], 15)
    np.testing.assert_allclose(outer[0], 30)
    np.testing.assert_allclose(inner[1], [10, np.sqrt(200), 20, np.sqrt(200)])
    np.testing.assert_allclose(outer[1], [25, np.sqrt(875), 35, np.sqrt(875)])
    rotated = circle_distances(np.array([[0., 5.]]), np.array([90.]), 15, angles)
    np.testing.assert_allclose(rotated[0], inner[1])


def test_rectangle_north_zero_and_rotated_geometry():
    xy = np.array([[35., 35.], [6., 6.]])
    values = rectangle_distances(xy, np.array([180., 0.]))
    np.testing.assert_allclose(values[0], values[1], atol=1e-12)
    np.testing.assert_allclose(rectangle_distances(np.array([[20.5, 10.]]), np.zeros(1), np.array([0.,90.,180.,270.]))[0], [31,20.5,10,20.5])


def test_joint_frame_counts_preserve_filtering_ties_and_unit_order():
    times = np.array([0., 1., 3.])
    spikes = np.array([-1., 0., .5, 1., 2., 2.1, 3., 4.])
    clusters = np.array([7, 9, 7, 99, 9, 7, 9, 7])
    units, counts = frame_counts(spikes, clusters, [9, 11, 7], times)
    np.testing.assert_array_equal(units, [7, 9])
    selected = (spikes >= times[0]) & (spikes <= times[-1])
    expected = np.array([
        spatial.count_spikes_by_frame(spikes[selected & (clusters == unit)], times)
        for unit in units
    ], dtype=np.int32)
    np.testing.assert_array_equal(counts, expected)
    assert counts.dtype == np.int32


def test_cached_permutation_ranks_preserve_seeded_p_and_bootstrap_ties():
    x = np.array([1., 1., 2., 2., 4., 4., 8., 8.])
    bands = ("0-8", "8-16", "16-", "all")
    y = np.array([[1, 1, 2, 3, 1, 1, 2, 3], [4, 3, 3, 2, 1, 1, 2, 2],
                  [5] * 8, [3, 3, 1, 1, 2, 2, 1, 1]], dtype=float)
    index = pd.MultiIndex.from_product([["A"], range(len(x))], names=["probe", "unit_id"])
    centers = pd.DataFrame({"rf_y_deg": x}, index=index)
    peaks = pd.DataFrame([
        ("A", unit, band, values[unit]) for band, values in zip(bands, y) for unit in range(len(x))
    ], columns=["probe", "unit_id", "band", "d_peak_cm"])
    permutations, bootstraps, seed = 199, 100, 9
    _, result, _ = distance_correlations(
        centers, peaks, permutations=permutations, bootstraps=bootstraps, seed=seed,
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(x), size=(bootstraps, len(x)))
    permuted_x = np.array([rng.permutation(x) for _ in range(permutations)])
    for band, values in zip(bands, y):
        rho = float(row_spearman(x, values))
        null = row_spearman(permuted_x, np.broadcast_to(values, permuted_x.shape))
        p = (1 + np.count_nonzero(np.abs(null) >= abs(rho))) / (permutations + 1) if np.isfinite(rho) else np.nan
        ci, fraction = bootstrap_bounds(row_spearman(x[samples], values[samples]))
        np.testing.assert_equal(result.loc[band, "p"], p)
        np.testing.assert_array_equal(result.loc[band, ["ci_low", "ci_high"]], ci)
        assert result.loc[band, "bootstrap_defined_fraction"] == fraction


def test_histogram_endpoints_distance_bands_and_unvisited_cells():
    projection = ray_histogram_matrix(np.array([[0.,30.], [30.,60.]]), np.array([0.,30.,60.]))
    np.testing.assert_array_equal((projection @ np.array([1.,2.])).reshape(2,2), [[3,0],[1,2]])
    centers = np.array([0.,8.,16.,30.,60.])
    np.testing.assert_array_equal(band_mask(centers, None, 8), [1,1,0,0,0])
    np.testing.assert_array_equal(band_mask(centers, 8, 16), [0,0,1,0,0])
    np.testing.assert_array_equal(band_mask(centers, 30, 60), [0,0,0,0,1])
    rates = rates_from_counts(np.ones((1,2,2)), np.array([[1.,0.],[1.,0.]]), sigma=1)
    assert np.isnan(rates[:,:,1]).all()


def test_peak_distance_uses_full_map_coordinates(tmp_path):
    values = np.zeros((1,4,6))
    values[0,2,1] = 8
    values[0,1,3] = 10
    values[0,3,5] = 12
    path = tmp_path / "map.rfmap"
    spatial.save_egocentric_rfmap(path, values,
        {"distance_edges": np.array([0,4,8,12,16,20,24]), "theta_edges": np.arange(0,361,90)}, [27], [0,1])
    peaks = ebc_distance_peaks(path)
    np.testing.assert_array_equal(peaks.d_peak_cm, [6,14,22,22])
    np.testing.assert_array_equal(peaks.bearing_deg, [225,135,315,315])


def test_correlations_share_units_and_handle_constant_band():
    units = np.arange(20)
    index = pd.MultiIndex.from_product([["A"], units], names=["probe","unit_id"])
    centers = pd.DataFrame({"rf_y_deg": units}, index=index)
    rows = [("A", int(u), band, d) for u in units for band,d in
            [("0-8", u/3), ("8-16", 15-u/3), ("16-", 20), ("all", float(u))]]
    peaks = pd.DataFrame(rows, columns=["probe","unit_id","band","d_peak_cm"])
    data, stats, differences = distance_correlations(centers, peaks, permutations=199, bootstraps=100)
    assert len(data) == 20
    np.testing.assert_allclose(stats.loc[["0-8","8-16"],"rho"], [1,-1])
    assert np.isnan(stats.loc["16-","rho"])
    assert np.isnan(stats.loc["16-","p"])
    assert stats.loc["0-8","q"] < .05
    assert differences.iloc[0].ci_low > 1.9
    assert np.isnan(row_spearman([1,1,1], [1,2,3]))


def test_discrete_bootstrap_does_not_hide_undefined_resamples():
    interval, fraction = bootstrap_bounds(np.r_[np.full(60, -.2), np.full(40, np.nan)])
    np.testing.assert_array_equal(interval, [-1, 1])
    assert fraction == .6
    interval, _ = bootstrap_bounds(np.r_[np.full(60, -.6), np.full(40, np.nan)], limit=2)
    np.testing.assert_array_equal(interval, [-2, 2])


def test_missing_band_uses_identical_complete_units():
    index = pd.MultiIndex.from_product([["A"], [1, 2, 3, 4]], names=["probe", "unit_id"])
    centers = pd.DataFrame({"rf_y_deg": [1, 2, 3, 4]}, index=index)
    rows = [("A", u, band, u) for u in range(1, 5) for band in ("0-8", "8-16", "16-", "all")
            if (u, band) != (3, "8-16")]
    peaks = pd.DataFrame(rows, columns=["probe", "unit_id", "band", "d_peak_cm"])
    data, stats, _ = distance_correlations(centers, peaks, permutations=29, bootstraps=30)
    assert data.index.tolist() == [("A", 1), ("A", 2), ("A", 4)]
    assert (stats.n == 3).all()


def test_spatial_information_normalizes_the_smoothed_map():
    occupancy = np.arange(441, dtype=float)
    np.testing.assert_allclose(spatial_information(occupancy[None, :] * 5, occupancy), 0, atol=1e-12)
    spikes = np.zeros((1, 441))
    spikes[0, 220] = 100
    assert spatial_information(spikes, occupancy)[0] > 0


def test_behavior_matching_and_block_weighted_rates():
    n = 400
    xy = np.tile([[35.,35.],[6.,6.],[35.,6.],[6.,35.]], (n//4,1))
    hd = np.tile([180.,0.,0.,180.], n//4)
    data = prepare_controls(dict(times=np.arange(n)*.1,xy=xy,hd=hd,unit_ids=np.array([7]),counts=np.full((1,n),2),dt=.1))
    states = behavior_states(data)
    pairs = match_states(states, mode="geometry")
    assert len(pairs) == 2
    np.testing.assert_allclose(pairs.geometry_rms_cm, 0, atol=1e-12)
    assert len(set(pairs.state_a) | set(pairs.state_b)) == 4
    rates = matched_firing_rates(data, states, pairs, bootstraps=30, block_s=2)
    np.testing.assert_allclose(rates.loc[7,["rate_a_hz","rate_b_hz"]], [20,20])
    assert abs(rates.loc[7,"difference_hz"]) < 1e-12


def test_hd_lookup_does_not_train_on_test_spikes():
    n=2000
    hd=np.arange(n)%60*6.
    data=dict(times=np.arange(n)*.1,dt=.1,hd=hd,counts=np.ones((1,n)))
    first=hd_expected_counts(data,folds=2,guard_s=0)
    data["counts"][0,:n//2]=100
    second=hd_expected_counts(data,folds=2,guard_s=0)
    np.testing.assert_allclose(first[:,:n//2],second[:,:n//2])
    assert (second[:,n//2:] > first[:,n//2:]).all()


def test_hd_only_and_planted_ebc_controls():
    rng=np.random.default_rng(12)
    n=8000
    xy=rng.uniform(1,40,(n,2)); hd=rng.uniform(0,360,n)
    distances=rectangle_distances(xy,hd)
    hd_rate=8*np.exp(1.2*np.cos(np.deg2rad(hd)))
    ebc_rate=1+90*np.exp(-.5*((distances[:,0]-5)/2)**2)
    counts=rng.poisson(np.array([hd_rate,ebc_rate])*.1)
    data=prepare_controls(dict(times=np.arange(n)*.1,xy=xy,hd=hd,unit_ids=np.array([7,9]),counts=counts,dt=.1))
    result=hd_only_controls(data,simulations=39,batch_size=13,workers=1,seed=3)
    assert result["table"].loc[9,"mvl_0-8"] > np.quantile(result["null_strength"][1,:,0],.975)
    assert result["table"].loc[7,"p_0-8"] > .05
    assert result["table"].hd_prediction_coverage.eq(1).all()
