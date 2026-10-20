import numpy as np
import pytest

from Utils.theta_analysis import (
    analyze_theta_units,
    extract_theta_phase,
    phase_locking,
    phase_precession,
    sample_theta_phase,
    summarize_lfp_reference,
    theta_rhythmicity,
)


def ideal_theta(times, frequency=8., phase_offset=0., segment_ids=None):
    times = np.asarray(times)
    return dict(times_s=times,
                unwrapped_phase_rad=2 * np.pi * frequency * times + phase_offset,
                segment_id=np.zeros(len(times), dtype=int) if segment_ids is None else segment_ids)


def test_acg_excludes_self_pairs_and_does_not_bridge_epochs():
    result = theta_rhythmicity([.05, .15, 1.05, 1.15], [[0., .2], [1., 1.2]])
    counts, lags = result["acg_counts"], result["acg_lags_s"]
    assert counts.sum() == 4
    assert counts[np.argmin(abs(lags))] == 0
    assert counts[np.argmin(abs(lags - .1))] == 2
    assert result["n_spikes"] == 4
    # Boundary spikes are assigned only to the right-hand half-open epoch.
    one = theta_rhythmicity([1.], [[0., 1.], [1., 2.]])
    assert one["n_spikes"] == 1
    assert one["acg_counts"].sum() == 0
    assert np.isnan(one["theta_index"])


def test_rhythmic_population_has_peak_near_eight_hz():
    rng = np.random.default_rng(42)
    # Gaussian timing jitter damps high harmonics while preserving 8-Hz firing.
    spikes = np.sort(np.arange(1., 121., .125) + rng.normal(0, .012, 960))
    result = theta_rhythmicity(spikes, [[0., 122.]])
    assert abs(result["peak_frequency_hz"] - 8) < .3
    assert result["theta_index"] >= 5
    assert result["is_theta_rhythmic"]
    assert result["parameters"]["spectrum_convention"] == "magnitude"


def test_tonic_poisson_has_no_strong_theta_peak():
    rng = np.random.default_rng(10)
    spikes = np.cumsum(rng.exponential(.03, 4000))
    result = theta_rhythmicity(spikes, [[0., spikes[-1] + 1.]])
    assert not result["is_theta_rhythmic"]


def test_hilbert_recovers_known_phase_and_excludes_gaps_and_edges():
    times = np.r_[np.arange(0, 12, .002), np.arange(14, 26, .002)]
    voltage = np.cos(2 * np.pi * 8 * times)
    theta = extract_theta_phase(times, voltage, sample_rate_hz=500, edge_exclusion_s=2)
    selected = ((times > 3) & (times < 9)) | ((times > 17) & (times < 23))
    error = np.angle(np.exp(1j * (theta["phase_rad"][selected] - 2 * np.pi * 8 * times[selected])))
    assert np.max(abs(error)) < .1
    assert len(theta["support_intervals_s"]) == 2
    test_spikes = np.array([.5, 4., 11.5, 13., 14.5, 18., 25.5])
    sampled = sample_theta_phase(test_spikes, theta)
    np.testing.assert_array_equal(sampled["valid_mask"], [False, True, False, False, False, True, False])
    assert np.isnan(sampled["phase_rad"][3])


@pytest.mark.parametrize("constant", [0., 500., 1.e8])
def test_constant_lfp_has_no_measurable_phase_or_spurious_locking(constant):
    times = np.arange(0., 8., .002)
    theta = extract_theta_phase(times, np.full_like(times, constant), sample_rate_hz=500)
    assert not theta["valid_mask"].any()
    assert theta["support_intervals_s"].shape == (0, 2)
    assert theta["phase_qc"]["constant_run_count"] == 1
    locked = phase_locking(np.arange(1., 7., .125), theta)
    assert locked["n_spikes"] == 0
    assert np.isnan(locked["mean_vector_length"])
    assert np.isnan(locked["rayleigh_p"])
    assert not locked["is_phase_locked"]


def test_numerically_zero_hilbert_amplitude_splits_phase_support(monkeypatch):
    times = np.arange(0., 8., .002)
    voltage = np.cos(2 * np.pi * 8 * times)
    analytic = np.exp(2j * np.pi * 8 * times)
    analytic[(times >= 3.) & (times < 4.)] *= 1.e-30
    monkeypatch.setattr("Utils.theta_analysis.signal.hilbert", lambda values: analytic.copy())
    theta = extract_theta_phase(times, voltage, sample_rate_hz=500, edge_exclusion_s=1.)
    assert len(theta["support_intervals_s"]) == 2
    assert not theta["valid_mask"][(times >= 3.) & (times < 4.)].any()
    sampled = sample_theta_phase([2., 3.5, 5.], theta)
    np.testing.assert_array_equal(sampled["valid_mask"], [True, False, True])
    assert theta["phase_qc"]["runs"][0]["below_numerical_floor_samples"] == 500


def test_numerical_amplitude_floor_is_scale_relative_not_physiological():
    times = np.arange(0., 8., .002)
    theta = extract_theta_phase(times, 1.e-20 * np.cos(2 * np.pi * 8 * times), sample_rate_hz=500)
    assert theta["valid_mask"].any()
    assert np.nanmedian(theta["amplitude"]) > 0
    assert theta["parameters"]["physiological_amplitude_threshold"] is None


def test_phase_locking_recovers_direction_occupancy_and_gap_exclusions():
    times = np.r_[np.arange(0., 10., .001), np.arange(12., 22., .001)]
    segment_ids = np.r_[np.zeros(10000, dtype=int), np.ones(10000, dtype=int)]
    theta = ideal_theta(times, segment_ids=segment_ids)
    spikes = np.sort(np.r_[np.arange(.03125, 10., .125), 11., np.arange(12.03125, 22., .125)])
    result = phase_locking(spikes, theta)
    assert abs(result["preferred_phase_rad"] - np.pi / 2) < 1e-8
    assert result["mean_vector_length"] > .999
    assert result["rayleigh_p"] < .01
    assert result["is_phase_locked"]
    assert result["n_spikes_excluded"] == 1
    assert result["phase_counts"].sum() == 160
    assert result["phase_occupancy_s"].sum() == pytest.approx(19.998)
    assert np.all(result["phase_occupancy_s"] > 0)


def test_phase_histogram_retains_missing_vs_observed_zero():
    theta = ideal_theta(np.arange(0., .02, .001), frequency=1.)
    result = phase_locking([], theta)
    missing = result["phase_occupancy_s"] == 0
    assert missing.any()
    assert np.all(np.isnan(result["phase_rate_hz"][missing]))
    assert np.all(result["phase_rate_hz"][~missing] == 0)
    assert np.isnan(result["rayleigh_p"])


def test_phase_exposure_conditions_both_spikes_and_occupancy():
    theta = ideal_theta(np.arange(0., 4., .001), frequency=8.)
    spikes = np.arange(.03125, 4., .125)
    result = phase_locking(spikes, theta, exposure_intervals_s=[[1., 2.]])
    assert result["n_spikes"] == 8
    assert result["n_spikes_excluded"] == 24
    assert result["phase_counts"].sum() == 8
    assert result["phase_occupancy_s"].sum() == pytest.approx(1.)
    assert np.all(np.isnan(result["spike_phase_rad"][~result["spike_valid_mask"]]))
    empty = phase_locking(spikes, theta, exposure_intervals_s=[])
    assert empty["n_spikes"] == 0
    assert empty["phase_occupancy_s"].sum() == 0
    assert np.all(np.isnan(empty["phase_rate_hz"]))


def test_phase_acg_distinguishes_faster_from_slower_than_theta():
    times = np.arange(0., 120., .002)
    theta = ideal_theta(times, frequency=8.)
    fast = phase_precession(np.arange(.5, 119., 1 / 8.8), theta)
    slow = phase_precession(np.arange(.5, 119., 1 / 7.2), theta)
    assert fast["eligible"] and slow["eligible"]
    assert fast["peak_frequency_per_cycle"] == pytest.approx(1.1, abs=.03)
    assert slow["peak_frequency_per_cycle"] == pytest.approx(.9, abs=.03)
    assert fast["is_phase_precessing"]
    assert not slow["is_phase_precessing"]
    assert not fast["parameters"]["exact_published_replication"]


def test_exactly_phase_locked_spikes_are_not_precession_candidates():
    times = np.arange(0., 60., .002)
    theta = ideal_theta(times, frequency=8.)
    result = phase_precession(np.arange(.5, 59., .125), theta)
    assert result["eligible"]
    assert result["peak_frequency_per_cycle"] == pytest.approx(1., abs=.03)
    assert not result["precession_candidate"]
    assert not result["is_phase_precessing"]


def test_phase_acg_rejects_nonadvancing_lfp_phase_for_classification():
    times = np.arange(0., 10., .002)
    theta = ideal_theta(times)
    theta["unwrapped_phase_rad"][2000] -= 1
    result = phase_precession(np.arange(.5, 9.5, 1 / 8.8), theta, require_monotonic_theta=True)
    assert not result["theta_phase_monotonic"]
    assert not result["eligible"]
    assert not result["is_phase_precessing"]


@pytest.mark.parametrize("spikes", [[2., 1.], [1., np.nan], [[1., 2.]]])
def test_invalid_spike_timestamps_are_rejected(spikes):
    with pytest.raises(ValueError, match="Spike timestamps"):
        theta_rhythmicity(spikes, [[0., 10.]])


def test_invalid_intervals_and_irregular_lfp_clock_are_rejected():
    with pytest.raises(ValueError, match="nonoverlapping"):
        theta_rhythmicity([1., 2.], [[0., 2.], [1., 3.]])
    with pytest.raises(ValueError, match="regular"):
        extract_theta_phase([0., .002, .0045, .006], [1., 2., 3., 4.], sample_rate_hz=500)
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_theta_phase([0., .002, .002, .006], [1., 2., 3., 4.])


def test_unit_analysis_is_independent_and_corrects_only_assessed_phase_tests():
    times = np.arange(0., 28., .002)
    theta = extract_theta_phase(times, np.cos(2 * np.pi * 8 * times), sample_rate_hz=500.)
    spikes = {
        "locked": np.arange(1.03125, 25., .125),
        "dispersed": np.linspace(1., 25., 210),
        "sparse": np.linspace(1., 25., 100),
        "lfp_edge": np.full(101, .01),
    }
    result = analyze_theta_units(
        spikes, theta, tracking_intervals_s=[[0., 28.]], moving_intervals_s=[[0., 28.]],
    )
    table = result["summary"]
    assert table.loc["locked", "is_phase_locked"]
    assert not table.loc["dispersed", "is_phase_locked"]
    assert table.loc["sparse", "theta_status"] == "insufficient_spikes"
    assert table.loc["lfp_edge", "phase_locking_status"] == "insufficient_phase_spikes"
    assert table.loc[["sparse", "lfp_edge"], "rayleigh_q_bh"].isna().all()
    p_locked = table.loc["locked", "rayleigh_p"]
    p_dispersed = table.loc["dispersed", "rayleigh_p"]
    np.testing.assert_allclose(
        table.loc[["locked", "dispersed"], "rayleigh_q_bh"],
        [min(2 * p_locked, p_dispersed), p_dispersed], rtol=1e-12, atol=0,
    )

    below_coverage = analyze_theta_units(
        spikes, theta, tracking_intervals_s=[[0., 28.]], moving_intervals_s=[[0., 28.]],
        spatial_coverage=.8,
    )["summary"]
    assert np.isfinite(below_coverage.loc["locked", "theta_index"])
    assert below_coverage["is_theta_rhythmic"].isna().all()
    assert below_coverage["is_phase_locked"].isna().all()
    assert below_coverage["precession_candidate"].isna().all()
    assert below_coverage["rayleigh_q_bh"].isna().all()


def test_empty_unit_analysis_returns_complete_typed_summary():
    result = analyze_theta_units(
        {}, dict(parameters={}, phase_qc={}),
        tracking_intervals_s=[], moving_intervals_s=[],
    )
    table = result["summary"]
    assert table.empty
    assert table.index.name == "unit_id"
    assert result["units"] == {}
    assert table["unit_id"].dtype == object
    assert table["n_moving_spikes"].dtype == np.dtype("int64")
    assert table["rayleigh_q_bh"].dtype == np.dtype("float64")
    assert str(table["is_phase_locked"].dtype) == "boolean"
    assert table["theta_status"].dtype == object


def test_lfp_reference_without_movement_retains_spectrum_and_qc():
    times = np.arange(0., 8., .002)
    voltage = np.cos(2 * np.pi * 8 * times)
    theta = extract_theta_phase(times, voltage, sample_rate_hz=500.)
    reference = summarize_lfp_reference(
        dict(times_s=times, voltage_uv=voltage, sample_rate_hz=500.), theta, [],
    )
    assert reference["summary"]["valid_lfp_seconds"] > 0
    assert reference["summary"]["power_4_11_over_1_40"] > .99
    assert len(reference["spectrum"]["frequency_hz"]) > 0
    assert np.isfinite(reference["spectrum"]["psd_uv2_per_hz"]).all()
    trace = reference["trace"]
    assert trace["status"] == "no_movement"
    assert np.isnan(trace["midpoint_s"])
    for key in ("time_from_midpoint_s", "voltage_uv", "filtered_uv", "amplitude_uv"):
        assert trace[key].shape == (0,)
